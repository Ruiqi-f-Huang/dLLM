#!/usr/bin/env python3
"""Single-device online MDLM diffu-GRPO pilot and candidate-group audit."""
import argparse
import ast
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _reward_worker(connection, text, answer, timeout):
    import resource
    import math
    import sandbox_gsm8k
    try:
        cpu_limit = max(1, math.ceil(timeout))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        connection.send(bool(sandbox_gsm8k.evaluate_samples(text, answer, timeout)))
    except Exception:
        connection.send(False)
    finally:
        connection.close()


def execution_reward(text, answer, timeout):
    # Process isolation and wall-clock timeout; this is not an OS security sandbox.
    ctx = mp.get_context('spawn')
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_reward_worker,
                          args=(sender, text, answer, timeout), daemon=True)
    process.start()
    sender.close()
    try:
        return float(receiver.recv()) if receiver.poll(timeout + 3) else 0.0
    except EOFError:
        return 0.0
    finally:
        if process.is_alive():
            process.terminate()
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join()
        receiver.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default=str(ROOT / 'data/gsm8k_pot_result/sft_train.jsonl'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--limit', type=int, default=256)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--group-size', type=int, default=8)
    parser.add_argument('--microbatch', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--nfe', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--prompt-mask', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--beta', type=float, default=0.04)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--timeout', type=float, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--save-at', type=int, nargs='*', default=[])
    args = parser.parse_args()
    if (min(args.limit, args.updates, args.microbatch, args.iterations,
            args.nfe, args.save_every) < 1 or args.group_size < 2
            or args.temperature <= 0 or args.lr <= 0 or args.beta < 0
            or not 0 < args.epsilon < 1 or not 0 <= args.prompt_mask <= 1
            or args.timeout <= 0 or any(x < 1 or x > args.updates for x in args.save_at)):
        parser.error('Invalid training or sampling parameters.')

    import torch
    import lightning as L
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import algo
    import dataloader
    import main as main_module
    from diffu_grpo import (completion_mask, grouped_advantages, proxy_logps,
                           reference_adapter, clipped_loss)
    if not torch.cuda.is_available():
        raise RuntimeError('This pilot requires CUDA and the s-flm training environment.')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Never silently append a different run or overwrite a checkpoint.
    manifest = output / 'run.json'
    with manifest.open('x') as handle:
        json.dump(vars(args), handle, indent=2)
    L.seed_everything(args.seed)
    rng = random.Random(args.seed)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'configs')):
        config = compose(config_name='sft-a800-lora-control', overrides=[
            f'training.finetune_path={Path(args.checkpoint).resolve()}',
            'training.finetune_weights=ema', 'training.ema=0',
            'lora.dropout=0.0'])
    OmegaConf.save(config, output / 'config.yaml', resolve=False)
    tokenizer = dataloader.get_tokenizer(config)
    model = main_module._initialize_finetune(algo.MDLM, config, tokenizer).cuda()
    # eval() disables backbone dropout but keeps autograd enabled during updates.
    model.eval()
    model.sampler.temperature = args.temperature
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty training data.')
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('sample_id must be unique.')
    rng.shuffle(rows)
    rows = rows[:args.limit]
    separator = tokenizer(str(config.data.separator), add_special_tokens=False).input_ids
    prefixes = {}
    for row in rows:
        prefix = ([tokenizer.bos_token_id]
                  + tokenizer(row['question'].strip(), add_special_tokens=False).input_ids
                  + separator)
        if len(prefix) >= model.num_tokens:
            raise ValueError(f"No completion space for {row['sample_id']}")
        prefixes[row['sample_id']] = prefix
    (output / 'selected_ids.json').write_text(json.dumps([r['sample_id'] for r in rows]))

    def save(step):
        torch.save({'state_dict': model.state_dict(),
                    'hyper_parameters': dict(model.hparams),
                    'pytorch-lightning_version': L.__version__,
                    'global_step': step, 'grpo_args': vars(args),
                    'optimizer': optimizer.state_dict()}, output / f'step-{step:06d}.ckpt')

    step = 0
    group_stats = []
    groups = len(rows) if args.audit_only else (args.updates + args.iterations - 1) // args.iterations
    with (output / 'rollouts.jsonl').open('x') as rollouts, (output / 'metrics.jsonl').open('x') as metrics:
        for group_index in range(groups):
            row = rows[group_index % len(rows)]
            prefix = torch.tensor([prefixes[row['sample_id']]], device='cuda')
            lengths = torch.full((args.group_size,), prefix.shape[1], device='cuda', dtype=torch.long)
            prefix = prefix.expand(args.group_size, -1)
            chunks = []
            with torch.no_grad():
                for start in range(0, args.group_size, args.microbatch):
                    end = min(start + args.microbatch, args.group_size)
                    sample, _ = model.generate_samples(
                        num_samples=end-start, num_steps=args.nfe,
                        prefix_tokens=prefix[start:end], prefix_lengths=lengths[start:end])
                    chunks.append(sample)
            tokens = torch.cat(chunks)
            valid = completion_mask(tokens, lengths, tokenizer.eos_token_id,
                                    tokenizer.pad_token_id, model.mask_index)
            rewards, texts = [], []
            for i in range(args.group_size):
                ids = tokens[i, int(lengths[i]):].tolist()
                if tokenizer.eos_token_id in ids:
                    ids = ids[:ids.index(tokenizer.eos_token_id)]
                text = tokenizer.decode(ids, skip_special_tokens=True)
                reward = execution_reward(text, row['answer'], args.timeout)
                try:
                    ast.parse(text)
                    parses = True
                except SyntaxError:
                    parses = False
                texts.append(text)
                rewards.append(reward)
                rollouts.write(json.dumps({'group': group_index, 'sample_id': row['sample_id'],
                    'candidate': i, 'reward': reward, 'raw_parses': parses,
                    'text': text, 'tokens': ids}, ensure_ascii=False) + '\n')
            rollouts.flush()
            rewards = torch.tensor(rewards, device='cuda')
            advantages = grouped_advantages(rewards, args.group_size)
            stats = {'group': group_index, 'sample_id': row['sample_id'],
                     'reward': rewards.mean().item(), 'pass_any': bool(rewards.any()),
                     'mixed': bool((rewards.max() - rewards.min()).item()),
                     'unique_fraction': len(set(texts)) / args.group_size}
            group_stats.append(stats)
            if not args.audit_only:
                seeds = [rng.randrange(2**31) for _ in range(args.iterations)]
                cache = []
                with torch.no_grad():
                    for seed in seeds:
                        old, ref = [], []
                        for start in range(0, args.group_size, args.microbatch):
                            sl = slice(start, start + args.microbatch)
                            old.append(proxy_logps(model, tokens[sl], lengths[sl], args.prompt_mask, seed))
                            with reference_adapter(model):
                                ref.append(proxy_logps(model, tokens[sl], lengths[sl], args.prompt_mask, seed))
                        cache.append((torch.cat(old), torch.cat(ref)))
                for seed, (old, ref) in zip(seeds, cache):
                    optimizer.zero_grad(set_to_none=True)
                    totals = {'loss': 0.0, 'kl': 0.0, 'clip_fraction': 0.0}
                    count = valid.sum().item()
                    if count == 0:
                        raise ValueError('Rollout group has no valid tokens.')
                    for start in range(0, args.group_size, args.microbatch):
                        sl = slice(start, start + args.microbatch)
                        weight = valid[sl].sum().item() / count
                        if not weight:
                            continue
                        current = proxy_logps(model, tokens[sl], lengths[sl], args.prompt_mask, seed)
                        loss, diagnostics = clipped_loss(current, old[sl], ref[sl],
                            advantages[sl], valid[sl], args.epsilon, args.beta)
                        (loss * weight).backward()
                        totals['loss'] += loss.item() * weight
                        for key in diagnostics:
                            totals[key] += diagnostics[key] * weight
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 0.2, error_if_nonfinite=True)
                    optimizer.step()
                    step += 1
                    metrics.write(json.dumps(dict(stats, step=step, grad_norm=float(norm), **totals)) + '\n')
                    metrics.flush()
                    if step % args.save_every == 0 or step == args.updates or step in args.save_at:
                        save(step)
                    if step >= args.updates:
                        break
            else:
                metrics.write(json.dumps(stats) + '\n')
                metrics.flush()
            print(json.dumps(dict(stats, step=step)), flush=True)
    summary = {key: sum(float(s[key]) for s in group_stats) / len(group_stats)
               for key in ('reward', 'pass_any', 'mixed', 'unique_fraction')}
    summary.update(groups=len(group_stats), updates=step,
                   policy='raw', probability='full-mask proxy',
                   reward_protocol='existing evaluator recovery; first-EOS truncation')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

