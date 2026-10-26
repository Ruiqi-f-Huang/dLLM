#!/usr/bin/env python3
"""Paired validation of Base EMA and raw GRPO, with a seed reset after loading."""
import argparse
import ast
import gc
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--rl', required=True)
    parser.add_argument('--data', default=str(ROOT / 'data/gsm8k_pot_result/sft_validation.jsonl'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--rl-label', default='rl100')
    parser.add_argument('--expected-count', type=int, default=632)
    parser.add_argument('--temperature', type=float, default=.1)
    args = parser.parse_args()
    import lightning as L
    import torch
    from hydra import compose, initialize_config_dir
    import algo
    import dataloader
    import main as entry
    import sandbox_gsm8k
    from omegaconf import OmegaConf

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(x) for x in Path(args.data).read_text().splitlines() if x.strip()]
    if len(rows) != args.expected_count or len({r['sample_id'] for r in rows}) != args.expected_count:
        raise ValueError('Unexpected dataset size or duplicate sample IDs.')
    if args.rl_label == 'base' or args.batch_size < 1 or args.temperature <= 0:
        raise ValueError('Invalid label, batch size or temperature.')
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'configs')):
        config = compose(config_name='sft-a800-lora-control', overrides=[
            f'training.finetune_path={Path(args.base).resolve()}',
            'training.finetune_weights=ema', 'training.ema=0', 'lora.dropout=0.0'])
    tokenizer = dataloader.get_tokenizer(config)
    sep = tokenizer(str(config.data.separator), add_special_tokens=False).input_ids
    prefixes = [[tokenizer.bos_token_id] + tokenizer(r['question'].strip(), add_special_tokens=False).input_ids + sep for r in rows]
    if any(len(p) >= 512 for p in prefixes):
        raise ValueError('Prefix exceeds generation space.')
    runs = {}
    for label in ('base', args.rl_label):
        L.seed_everything(42)
        if label == 'base':
            model = entry._initialize_finetune(algo.MDLM, config, tokenizer)
        else:
            model = algo.MDLM.load_from_checkpoint(args.rl, map_location='cpu', weights_only=False)
        model = model.cuda().eval()
        model.sampler.temperature = args.temperature
        for seed in (1, 2):
            # Model/adapter construction must not alter paired sampling RNG.
            L.seed_everything(seed)
            records = []
            for start in range(0, len(rows), args.batch_size):
                stop = min(start + args.batch_size, len(rows))
                batch_prefix, lengths = entry._pad_prefix_batch(
                    [torch.tensor(p, dtype=torch.long) for p in prefixes[start:stop]], model.device)
                with torch.no_grad():
                    tokens, _ = model.generate_samples(stop-start, num_steps=32,
                        prefix_tokens=batch_prefix, prefix_lengths=lengths)
                for j, row in enumerate(rows[start:stop]):
                    ids = tokens[j, int(lengths[j]):].cpu().tolist()
                    trimmed = ids[:ids.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in ids else ids
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    eos_text = tokenizer.decode(trimmed, skip_special_tokens=True)
                    def parses(code):
                        try:
                            ast.parse(code)
                            return True
                        except (SyntaxError, ValueError):
                            return False
                    records.append(dict(sample_id=row['sample_id'], question=row['question'],
                        response=text, eos_response=eos_text, tokens=ids,
                        correct=bool(sandbox_gsm8k.evaluate_samples(text, row['answer'], 2)),
                        eos_correct=bool(sandbox_gsm8k.evaluate_samples(eos_text, row['answer'], 2)),
                        raw_parses=parses(text), eos_parses=parses(eos_text)))
                print(json.dumps(dict(model=label, seed=seed, evaluated=stop,
                    correct=sum(r['correct'] for r in records))), flush=True)
            runs[f'{label}-seed{seed}'] = records
            (output / f'{label}-seed{seed}.json').write_text(json.dumps(records, ensure_ascii=False))
        del model
        gc.collect()
        torch.cuda.empty_cache()

    comparisons = []
    for seed in (1, 2):
        base, rl = runs[f'base-seed{seed}'], runs[f'{args.rl_label}-seed{seed}']
        assert [r['sample_id'] for r in base] == [r['sample_id'] for r in rl]
        for protocol, field in [('legacy-full', 'correct'), ('first-eos', 'eos_correct')]:
            gained = sum(not a[field] and b[field] for a,b in zip(base,rl))
            lost = sum(a[field] and not b[field] for a,b in zip(base,rl))
            n = gained + lost
            p = min(1., 2*sum(math.comb(n,k) for k in range(min(gained,lost)+1))/2**n) if n else 1.
            comparisons.append(dict(seed=seed, protocol=protocol,
                base_correct=sum(r[field] for r in base), rl_correct=sum(r[field] for r in rl),
                gained=gained, lost=lost, paired_exact_p=p,
                base_raw_parses=sum(r['raw_parses'] for r in base),
                rl_raw_parses=sum(r['raw_parses'] for r in rl),
                changed_outputs=sum(a['response'] != b['response'] for a,b in zip(base,rl))))
    summary = dict(protocol=dict(n=len(rows), nfe=32, temperature=args.temperature, sampler='ancestral',
        batch_size=args.batch_size, seeds=[1,2], base_weights='imported EMA',
        rl_weights='raw', rng_reset='after model initialization', max_length=512),
        rl_label=args.rl_label,
        comparisons=comparisons,
        limitations=['One training seed; exploratory pilot.',
                     'Generation seeds reuse the same questions, not independent datasets.',
                     'Legacy full output and EOS-truncated rewards are reported separately.'])
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()


