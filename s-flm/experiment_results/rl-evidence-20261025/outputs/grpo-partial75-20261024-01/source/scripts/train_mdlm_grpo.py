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


def _reward_worker(connection, text, answer, timeout, details=False):
    import resource
    import math
    import sandbox_gsm8k
    try:
        cpu_limit = max(1, math.ceil(timeout * (3 if details else 1)))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        connection.send(execution_details_local(text, answer, timeout) if details else
                        bool(sandbox_gsm8k.evaluate_samples(text, answer, timeout)))
    except Exception:
        connection.send(False)
    finally:
        connection.close()


def execution_reward(text, answer, timeout, details=False):
    # Process isolation and wall-clock timeout; this is not an OS security sandbox.
    ctx = mp.get_context('spawn')
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_reward_worker,
                          args=(sender, text, answer, timeout, details), daemon=True)
    process.start()
    sender.close()
    try:
        result = receiver.recv() if receiver.poll(timeout * (3 if details else 1) + 3) else False
        if details:
            return result if isinstance(result, dict) else {'infrastructure_failure': True, 'reward': 0.0}
        return float(result)
    except EOFError:
        return {'infrastructure_failure': True, 'reward': 0.0} if details else 0.0
    finally:
        if process.is_alive():
            process.terminate()
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join()
        receiver.close()


def execution_details_local(text, answer, timeout):
    import contextlib
    import io
    import sandbox_gsm8k as sandbox
    def strict(code):
        result=dict(parses=False, executes=False, numeric=False, correct=False)
        try:
            ast.parse(code);result['parses']=True
            with sandbox._time_limit(timeout):
                namespace=sandbox._safe_exec_no_timer(code)
                function=namespace.get('simple_math_problem')
                if function is None:return result
                stream=io.StringIO()
                with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(io.StringIO()):
                    value=function()
                result['executes']=True
                prediction=sandbox._to_number(value)
                if prediction is None:prediction=sandbox._extract_text_answer(stream.getvalue())
                result['numeric']=prediction is not None
                result['correct']=bool(sandbox._numbers_equal(prediction,sandbox._extract_gold_answer(answer)))
        except Exception:pass
        return result
    return dict(raw=strict(text),recovered=strict(sandbox._extract_code(text)),
                reward=float(sandbox.evaluate_samples(text,answer,timeout)),infrastructure_failure=False)


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
    parser.add_argument('--completion-mask', type=float, default=1.0)
    parser.add_argument('--prompt-mask', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--beta', type=float, default=0.04)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--timeout', type=float, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--save-at', type=int, nargs='*', default=[])
    parser.add_argument('--reshuffle-cycles', action='store_true')
    parser.add_argument('--precision', choices=['bfloat16','float32'], default='bfloat16')
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--diagnostic-pairs')
    parser.add_argument('--gradient-at', type=int, nargs='*', default=[0,1,10,50,100,250])
    args = parser.parse_args()
    if (min(args.limit, args.updates, args.microbatch, args.iterations,
            args.nfe, args.save_every) < 1 or args.group_size < 2
            or args.temperature <= 0 or args.lr <= 0 or args.beta < 0
            or not 0 < args.epsilon < 1 or not 0 <= args.prompt_mask <= 1
            or not 0 < args.completion_mask <= 1
            or (args.completion_mask < 1 and (args.microbatch != args.group_size or args.diagnostic_pairs))
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
                           reference_adapter, clipped_loss, partial_completion_mask)
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
    config.backbone_compute_dtype = args.precision
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
    if args.monitor:
        with torch.no_grad():
            xt = torch.full((1,model.num_tokens),model.mask_index,device=model.device,dtype=torch.long)
            pr = next(iter(prefixes.values()));xt[0,:len(pr)] = torch.tensor(pr,device=model.device)
            _, alpha = model.noise(torch.ones(1,1,device=model.device))
            raw = model.backbone(None,xt,model._process_sigma(model._sigma_from_alphat(alpha)),None)
            expected = torch.float32 if args.precision == 'float32' else torch.bfloat16
            if raw.dtype != expected:raise ValueError(f'Unexpected raw output dtype: {raw.dtype}')
            (output/'precision.json').write_text(json.dumps(dict(raw_dtype=str(raw.dtype),
                parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)))
    (output / 'selected_ids.json').write_text(json.dumps([r['sample_id'] for r in rows]))

    def save(step):
        torch.save({'state_dict': model.state_dict(),
                    'hyper_parameters': dict(model.hparams),
                    'pytorch-lightning_version': L.__version__,
                    'global_step': step, 'grpo_args': vars(args),
                    'optimizer': optimizer.state_dict(),
                    'diagnostic_state': {'python_rng': random.getstate(), 'schedule_rng': rng.getstate(),
                        'numpy_rng': __import__('numpy').random.get_state(),
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                        'next_group': group_index + 1, 'completed_inner_updates': (step - 1) % args.iterations + 1,
                        'group_tokens': tokens.cpu(), 'group_lengths': lengths.cpu(),
                        'old_ref_cache': [(a.cpu(), b.cpu()) for a,b in cache],
                        'mask_seeds': seeds, 'advantages': advantages.cpu(),
                        'group_schedule': [rows[i]['sample_id'] for i in order]}}, output / f'step-{step:06d}.ckpt')

    step = 0
    group_stats = []
    groups = len(rows) if args.audit_only else (args.updates + args.iterations - 1) // args.iterations
    from grpo_overfit import balanced_indices
    order = (balanced_indices(len(rows), groups, args.seed + 1) if args.reshuffle_cycles
             else [i % len(rows) for i in range(groups)])
    (output / 'group_schedule.json').write_text(json.dumps([rows[i]['sample_id'] for i in order]))
    from grpo_monitor import ratio_metrics, gradient_parts, parameter_step, ast_signature
    monitor_log = (output / 'monitor.jsonl').open('x') if args.monitor else None
    gradient_log = (output / 'gradients.jsonl').open('x') if args.monitor else None
    visits = {}; mixed_visits = {}
    def emit_geometry(state_step, scope, data, old, ref, seed, group):
        t, l, a, v = data
        result, vector = gradient_parts(model, parameters, t, l, old, ref, a, v, seed, args)
        gradient_log.write(json.dumps(dict(state_step=state_step, scope=scope, group=group, **result))+'\n')
        gradient_log.flush()
        return vector
    fixed = None
    if args.monitor and args.diagnostic_pairs:
        records = [json.loads(x) for x in Path(args.diagnostic_pairs).read_text().splitlines()]
        ft = torch.tensor([r['tokens'] for r in records], device=model.device)
        fl = torch.tensor([len(prefixes[r['sample_id']]) for r in records], device=model.device)
        fa = torch.tensor([r['reward'] for r in records], device=model.device) - .5
        fv = completion_mask(ft,fl,tokenizer.eos_token_id,tokenizer.pad_token_id,model.mask_index)
        with torch.no_grad(), reference_adapter(model):
            fr = torch.cat([proxy_logps(model,ft[i:i+args.microbatch],fl[i:i+args.microbatch],0.,42)
                            for i in range(0,len(ft),args.microbatch)])
        fixed = (ft,fl,fa,fv,fr)
    def fixed_geometry(state_step):
        if fixed is None:return
        ft,fl,fa,fv,fr = fixed
        with torch.no_grad():
            old = torch.cat([proxy_logps(model,ft[i:i+args.microbatch],fl[i:i+args.microbatch],0.,42)
                             for i in range(0,len(ft),args.microbatch)])
        emit_geometry(state_step, 'fixed-pairs-old-equals-current', (ft,fl,fa,fv), old, fr, 42, None)
    with (output / 'rollouts.jsonl').open('x') as rollouts, (output / 'metrics.jsonl').open('x') as metrics:
        for group_index in range(groups):
            row = rows[order[group_index]]
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
            rewards, texts, execution_records = [], [], []
            for i in range(args.group_size):
                ids = tokens[i, int(lengths[i]):].tolist()
                if tokenizer.eos_token_id in ids:
                    ids = ids[:ids.index(tokenizer.eos_token_id)]
                text = tokenizer.decode(ids, skip_special_tokens=True)
                details = execution_reward(text, row['answer'], args.timeout, details=True) if args.monitor else None
                reward = details['reward'] if details else execution_reward(text, row['answer'], args.timeout)
                if details: execution_records.append(details)
                try:
                    ast.parse(text)
                    parses = True
                except SyntaxError:
                    parses = False
                texts.append(text)
                rewards.append(reward)
                rollouts.write(json.dumps({'group': group_index, 'sample_id': row['sample_id'],
                    'candidate': i, 'reward': reward, 'raw_parses': parses,
                    'text': text, 'tokens': ids, 'full_tokens': tokens[i].cpu().tolist(),
                    'eos_found': tokenizer.eos_token_id in tokens[i, int(lengths[i]):].tolist(),
                    'execution': details}, ensure_ascii=False) + '\n')
            rollouts.flush()
            rewards = torch.tensor(rewards, device='cuda')
            advantages = grouped_advantages(rewards, args.group_size)
            stats = {'group': group_index, 'sample_id': row['sample_id'],
                     'reward': rewards.mean().item(), 'pass_any': bool(rewards.any()),
                     'mixed': bool((rewards.max() - rewards.min()).item()),
                     'unique_fraction': len(set(texts)) / args.group_size}
            visits[row['sample_id']] = visits.get(row['sample_id'],0)+1
            mixed_visits[row['sample_id']] = mixed_visits.get(row['sample_id'],0)+int(stats['mixed'])
            stats.update(question_visits=visits[row['sample_id']],mixed_visits=mixed_visits[row['sample_id']])
            if args.monitor:
                signatures = [ast_signature(t) for t in texts]
                stats.update(mean_valid_tokens=valid.sum(1).float().mean().item(),
                    success_count=int(rewards.sum().item()),advantages=advantages.tolist(),
                    ast_unique_fraction=len(set(x for x in signatures if x is not None))/len(texts),
                    execution=execution_records)
            group_stats.append(stats)
            if not args.audit_only:
                seeds = [rng.randrange(2**31) for _ in range(args.iterations)]
                if args.completion_mask < 1:
                    seeds = [seeds[0]] * args.iterations
                    completion_valid = valid
                    valid = partial_completion_mask(valid, args.completion_mask, seeds[0])
                    stats.update(completion_tokens=int(completion_valid.sum()),
                                 scored_tokens=int(valid.sum()),mask_seed=seeds[0])
                cache = []
                with torch.no_grad():
                    for seed in seeds:
                        old, ref = [], []
                        for start in range(0, args.group_size, args.microbatch):
                            sl = slice(start, start + args.microbatch)
                            old.append(proxy_logps(model, tokens[sl], lengths[sl], args.prompt_mask, seed, args.completion_mask))
                            with reference_adapter(model):
                                ref.append(proxy_logps(model, tokens[sl], lengths[sl], args.prompt_mask, seed, args.completion_mask))
                        cache.append((torch.cat(old), torch.cat(ref)))
                for inner, (seed, (old, ref)) in enumerate(zip(seeds, cache),1):
                    policy_vector = None
                    data = (tokens,lengths,advantages,valid)
                    if args.monitor and step in args.gradient_at:
                        policy_vector = emit_geometry(step, 'online-group-old-policy', data, old, ref, seed, group_index)
                        fixed_geometry(step)
                    before = [p.detach().clone() for p in parameters] if args.monitor else None
                    all_current = old.detach().clone() if args.monitor else None
                    optimizer.zero_grad(set_to_none=True)
                    totals = {'loss': 0.0, 'kl': 0.0, 'clip_fraction': 0.0,
                              'policy_loss': 0.0, 'kl_penalty': 0.0}
                    count = valid.sum().item()
                    if count == 0:
                        raise ValueError('Rollout group has no valid tokens.')
                    for start in range(0, args.group_size, args.microbatch):
                        sl = slice(start, start + args.microbatch)
                        weight = valid[sl].sum().item() / count
                        if not weight:
                            continue
                        current = proxy_logps(model, tokens[sl], lengths[sl], args.prompt_mask, seed, args.completion_mask)
                        loss, diagnostics = clipped_loss(current, old[sl], ref[sl],
                            advantages[sl], valid[sl], args.epsilon, args.beta)
                        if args.monitor:all_current[sl] = current.detach()
                        (loss * weight).backward()
                        totals['loss'] += loss.item() * weight
                        for key in diagnostics:
                            totals[key] += diagnostics[key] * weight
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 0.2, error_if_nonfinite=True)
                    optimizer.step()
                    step += 1
                    if args.monitor:
                        current = all_current
                        ratios = ratio_metrics(current,old,ref,advantages,valid,args.epsilon,args.beta)
                        changes = parameter_step(parameters,before,norm,policy_vector)
                        monitor_log.write(json.dumps(dict(step=step,inner_update=inner,group=group_index,
                            ratios=ratios, probability_timing='before_update',parameter_timing='after_update',
                            candidate_scores=[dict(candidate=i,reward=float(rewards[i]),advantage=float(advantages[i]),
                                valid_tokens=int(valid[i].sum()),
                                current_mean_logp=current[i][valid[i]].mean().item(),
                                old_mean_logp=old[i][valid[i]].mean().item(),
                                reference_mean_logp=ref[i][valid[i]].mean().item()) for i in range(len(tokens))],**changes))+'\n');monitor_log.flush()
                        if ((totals['kl'] > .01 or totals['clip_fraction'] > .05) and step % 10 == 0):
                            emit_geometry(step, 'online-after-update-trigger', data, old, ref, seed, group_index)
                    metrics.write(json.dumps(dict(stats, step=step, inner_update=inner, grad_norm=float(norm), **totals)) + '\n')
                    metrics.flush()
                    if step % args.save_every == 0 or step == args.updates or step in args.save_at:
                        save(step)
                    if step >= args.updates:
                        break
            else:
                metrics.write(json.dumps(stats) + '\n')
                metrics.flush()
            print(json.dumps(dict(stats, step=step)), flush=True)
    if args.monitor:
        emit_geometry(step, 'online-final-reused-group', data, old, ref, seed, group_index)
        fixed_geometry(step)
        monitor_log.close();gradient_log.close()
    summary = {key: sum(float(s[key]) for s in group_stats) / len(group_stats)
               for key in ('reward', 'pass_any', 'mixed', 'unique_fraction')}
    summary.update(groups=len(group_stats), updates=step,
                   policy='raw', probability=f'completion-mask-{args.completion_mask:g} proxy',
                   reward_protocol='existing evaluator recovery; first-EOS truncation')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
