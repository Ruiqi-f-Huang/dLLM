#!/usr/bin/env python3
"""Prepare and measure fixed-question online RL overfitting, not generalization."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grpo_overfit import select_mixed_questions, stable_seed, summarize_samples


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def prepare(args):
    train, validation, audit = map(read_jsonl, (args.train, args.validation, args.audit))
    for key in ('sample_id', 'question'):
        if {str(r[key]).strip() for r in train} & {str(r[key]).strip() for r in validation}:
            raise ValueError(f'Train/validation overlap: {key}')
    manifest = json.loads(Path(args.audit).with_name('run.json').read_text())
    if not manifest.get('audit_only') or manifest['group_size'] != 8 or manifest['temperature'] != .3:
        raise ValueError('Expected the independent Base T=.3, G=8 audit.')
    rows = select_mixed_questions(train, audit, args.count, args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'questions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    metadata = dict(selection='hash order of eligible mixed-reward IDs; no difficulty ranking',
                    seed=args.seed, count=args.count, audit_manifest=manifest,
                    ids=[r['sample_id'] for r in rows],
                    source_sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                   for p in (args.train, args.validation, args.audit)})
    (output / 'selection.json').write_text(json.dumps(metadata, indent=2))


def load_model(base, checkpoint):
    import torch
    import lightning as L
    from hydra import compose, initialize_config_dir
    import algo
    import dataloader
    import main as entry
    if not torch.cuda.is_available():
        raise RuntimeError('Measurement requires the CUDA s-flm environment.')
    L.seed_everything(42)
    if checkpoint:
        model = algo.MDLM.load_from_checkpoint(checkpoint, map_location='cpu', weights_only=False)
        tokenizer = model.tokenizer
    else:
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'configs')):
            config = compose(config_name='sft-a800-lora-control', overrides=[
                f'training.finetune_path={Path(base).resolve()}',
                'training.finetune_weights=ema', 'training.ema=0', 'lora.dropout=0.0'])
        tokenizer = dataloader.get_tokenizer(config)
        model = entry._initialize_finetune(algo.MDLM, config, tokenizer)
    return model.cuda().eval(), tokenizer


def measure(args):
    import torch
    from diffu_grpo import completion_mask, proxy_logps
    from train_mdlm_grpo import execution_reward
    if args.repeats < 2 or args.batch_size < 1 or args.max_pair_draws < 2 or args.timeout <= 0:
        raise ValueError('Invalid draw or batch count.')
    if len(set(args.temperatures)) != len(args.temperatures) or any(t <= 0 for t in args.temperatures):
        raise ValueError('Temperatures must be distinct positive values.')
    if bool(args.make_pairs) == bool(args.pairs) or (args.make_pairs and args.checkpoint):
        raise ValueError('Base creates holdout pairs; checkpoint measurements read existing pairs.')
    rows = read_jsonl(args.data)
    if len(rows) != 32 or len({r['sample_id'] for r in rows}) != 32:
        raise ValueError('Expected exactly 32 unique fixed training questions.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'run.json').write_text(json.dumps(vars(args), indent=2))
    model, tokenizer = load_model(args.base, args.checkpoint)
    separator = tokenizer(str(model.config.data.separator), add_special_tokens=False).input_ids
    prefixes = {r['sample_id']: [tokenizer.bos_token_id]
                + tokenizer(r['question'].strip(), add_special_tokens=False).input_ids + separator for r in rows}
    if any(len(p) >= model.num_tokens for p in prefixes.values()):
        raise ValueError('Question exceeds generation space.')

    @torch.no_grad()
    def generate(row, count, temperature, phase, chunk):
        seed = stable_seed(phase, args.eval_seed, row['sample_id'], temperature, chunk)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        prefix = torch.tensor([prefixes[row['sample_id']]], device=model.device).expand(count, -1)
        lengths = torch.full((count,), prefix.shape[1], device=model.device, dtype=torch.long)
        model.sampler.temperature = temperature
        tokens, _ = model.generate_samples(count, num_steps=32,
                        prefix_tokens=prefix, prefix_lengths=lengths)
        records = []
        for ids in tokens.cpu().tolist():
            suffix = ids[len(prefixes[row['sample_id']]):]
            eos_found = tokenizer.eos_token_id in suffix
            if eos_found:
                suffix = suffix[:suffix.index(tokenizer.eos_token_id)]
            text = tokenizer.decode(suffix, skip_special_tokens=True)
            reward = execution_reward(text, row['answer'], args.timeout)
            records.append(dict(sample_id=row['sample_id'], seed=seed, text=text,
                                reward=reward, tokens=ids, eos_found=eos_found,
                                completion_length=len(suffix)))
        return records

    # A separate RNG namespace creates a fixed diagnostic set. These candidates
    # never enter online optimization and are distinct from measurement draws.
    if args.make_pairs:
        pairs = []
        with (output / 'pair_candidates.jsonl').open('x') as handle:
            for row in rows:
                found = {}
                for start in range(0, args.max_pair_draws, args.batch_size):
                    candidates = generate(row, min(args.batch_size, args.max_pair_draws-start),
                                          .3, 'holdout-proxy-pairs', start)
                    for record in candidates:
                        handle.write(json.dumps(record, ensure_ascii=False)+'\n')
                        found.setdefault(int(record['reward']), record)
                    handle.flush()
                    if len(found) == 2:
                        break
                if len(found) != 2:
                    raise RuntimeError(f"No independent success/failure pair for {row['sample_id']}; do not silently reselect.")
                pairs.extend([found[1], found[0]])
        (output / 'pairs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in pairs))
    else:
        pairs = read_jsonl(args.pairs)

    grouped = {}
    for record in pairs:
        if record['reward'] not in (0, 1):
            raise ValueError('Diagnostic rewards must be binary.')
        sid, reward = record['sample_id'], int(record['reward'])
        if sid not in prefixes or reward not in (0, 1) or reward in grouped.setdefault(sid, {}):
            raise ValueError('Invalid or duplicate diagnostic pair.')
        if record['tokens'][:len(prefixes[sid])] != prefixes[sid]:
            raise ValueError('Diagnostic prefix mismatch.')
        grouped[sid][reward] = record
    if set(grouped) != set(prefixes) or any(set(v) != {0, 1} for v in grouped.values()):
        raise ValueError('Need one successful and one failed program per fixed question.')

    proxy_records = []
    with torch.no_grad():
        for row in rows:
            sid = row['sample_id']
            scores = {}
            for reward, record in grouped[sid].items():
                tokens = torch.tensor([record['tokens']], device=model.device)
                if tokens.shape[1] != model.num_tokens:
                    raise ValueError('Diagnostic tokens must preserve the full original sample.')
                lengths = torch.tensor([len(prefixes[sid])], device=model.device)
                valid = completion_mask(tokens, lengths, tokenizer.eos_token_id,
                                        tokenizer.pad_token_id, model.mask_index)
                logps = proxy_logps(model, tokens, lengths, 0., 42)[valid]
                if not logps.numel() or not torch.isfinite(logps).all():
                    raise ValueError('Invalid proxy token scores.')
                scores[reward] = dict(mean_logp=logps.mean().item(), sum_logp=logps.sum().item(),
                                      valid_tokens=logps.numel())
            proxy_records.append(dict(sample_id=sid, success=scores[1], failure=scores[0],
                mean_logp_gap=scores[1]['mean_logp']-scores[0]['mean_logp'],
                sum_logp_gap=scores[1]['sum_logp']-scores[0]['sum_logp']))
    proxy = dict(per_question=proxy_records,
                 mean_gap=sum(r['mean_logp_gap'] for r in proxy_records)/32,
                 mean_sum_gap=sum(r['sum_logp_gap'] for r in proxy_records)/32,
                 warning='Fixed-pair proxy preference, not exact program likelihood or GRPO loss.')
    (output / 'proxy.json').write_text(json.dumps(proxy, indent=2))

    sampling = {}
    for temperature in args.temperatures:
        records = []
        with (output / f'samples-t{temperature:g}.jsonl').open('x') as handle:
            for row in rows:
                for start in range(0, args.repeats, args.batch_size):
                    candidates = generate(row, min(args.batch_size, args.repeats-start),
                                          temperature, 'fresh-evaluation', start)
                    for j, record in enumerate(candidates):
                        record.update(repeat=start+j, temperature=temperature)
                        handle.write(json.dumps(record, ensure_ascii=False)+'\n')
                        records.append(record)
                    handle.flush()
                print(json.dumps(dict(temperature=temperature, sample_id=row['sample_id'],
                                      finished_draws=len(records))), flush=True)
        summary = summarize_samples(records, args.repeats)
        summary.update(mean_completion_length=sum(r['completion_length'] for r in records)/len(records),
                       eos_fraction=sum(r['eos_found'] for r in records)/len(records))
        sampling[f'{temperature:g}'] = summary
    (output / 'sampling.json').write_text(json.dumps(sampling, indent=2))
    (output / 'COMPLETE').touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    prep = modes.add_parser('prepare')
    prep.add_argument('--train', default=str(ROOT / 'data/gsm8k_pot_result/sft_train.jsonl'))
    prep.add_argument('--validation', default=str(ROOT / 'data/gsm8k_pot_result/sft_validation.jsonl'))
    prep.add_argument('--audit', required=True, help='Existing Base audit/rollouts.jsonl; needs sibling run.json')
    prep.add_argument('--output', required=True)
    prep.add_argument('--count', type=int, default=32)
    prep.add_argument('--seed', type=int, default=42)
    measurement = modes.add_parser('measure')
    measurement.add_argument('--base', required=True)
    measurement.add_argument('--checkpoint')
    measurement.add_argument('--data', required=True)
    measurement.add_argument('--output', required=True)
    measurement.add_argument('--pairs')
    measurement.add_argument('--make-pairs', action='store_true')
    measurement.add_argument('--repeats', type=int, default=16)
    measurement.add_argument('--temperatures', type=float, nargs='+', default=[.3, .1])
    measurement.add_argument('--batch-size', type=int, default=8)
    measurement.add_argument('--max-pair-draws', type=int, default=64)
    measurement.add_argument('--eval-seed', type=int, default=20261024)
    measurement.add_argument('--timeout', type=float, default=2.)
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else measure)(args)


if __name__ == '__main__':
    main()
