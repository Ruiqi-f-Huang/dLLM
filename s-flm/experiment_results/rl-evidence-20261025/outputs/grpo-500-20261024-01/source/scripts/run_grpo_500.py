#!/usr/bin/env python3
"""Run the fixed 500-update pilot, checks and validation without supervision."""
import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', default=str(ROOT / 'checkpoints/tinygsm/mdlm.ckpt'))
    args = parser.parse_args()
    run = Path(args.output).resolve()
    run.mkdir(parents=True, exist_ok=True)
    manifest = run / 'pipeline.json'
    with manifest.open('x') as handle:
        json.dump(vars(args), handle, indent=2)
    def stage(name):
        (run / 'STAGE').write_text(name + '\n')
        print(name, flush=True)
    def execute(command, log):
        with (run / log).open('x') as handle:
            subprocess.run([sys.executable, '-u'] + command, cwd=ROOT,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
    try:
        stage('preflight')
        train_path = ROOT / 'data/gsm8k_pot_result/sft_train.jsonl'
        valid_path = ROOT / 'data/gsm8k_pot_result/sft_validation.jsonl'
        train = [json.loads(x) for x in train_path.read_text().splitlines() if x.strip()]
        valid = [json.loads(x) for x in valid_path.read_text().splitlines() if x.strip()]
        assert len(train) == 5682 and len(valid) == 632
        for key in ('sample_id', 'question'):
            a = [str(r[key]).strip() for r in train]
            b = [str(r[key]).strip() for r in valid]
            assert len(set(a)) == len(a) and len(set(b)) == len(b)
            assert not set(a) & set(b), key
        shuffled = train.copy()
        random.Random(42).shuffle(shuffled)
        probe = shuffled[:32]
        (run / 'seen_train_probe32.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in probe))
        sources = list(ROOT.glob('*.py'))
        for directory in ('models', 'configs', 'callbacks'):
            sources += [p for p in (ROOT / directory).rglob('*')
                        if p.is_file() and p.suffix in ('.py', '.yaml')]
        sources += [ROOT / 'scripts' / name for name in
                    ('train_mdlm_grpo.py', 'eval_grpo_pilot.py', 'run_grpo_500.py')]
        hashes = {}
        for p in sources + [train_path, valid_path]:
            rel = p.relative_to(ROOT)
            target = run / 'source' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            hashes[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
        (run / 'source_sha256.json').write_text(json.dumps(hashes, indent=2))
        stage('train')
        execute(['scripts/train_mdlm_grpo.py', '--checkpoint', args.checkpoint,
                 '--data', str(train_path), '--output', str(run / 'train'),
                 '--limit', '5682', '--updates', '500', '--iterations', '2',
                 '--group-size', '8', '--microbatch', '8', '--nfe', '32',
                 '--temperature', '.3', '--seed', '42', '--save-every', '500',
                 '--save-at', '100', '250', '500'], 'train.log')
        stage('parameter_checks')
        import torch
        import algo
        import main as entry
        # Reference import resets optimizer and uses the Base EMA weights.
        check = torch.load(run / 'train/step-000100.ckpt', map_location='cpu', weights_only=False)
        cfg = check['hyper_parameters']['config']
        tokenizer = check['hyper_parameters']['tokenizer']
        base = entry._initialize_finetune(algo.MDLM, cfg, tokenizer)
        frozen = {n: t for n, t in base.state_dict().items()
                  if not n.endswith(('lora_A', 'lora_B'))}
        del check
        metrics = [json.loads(x) for x in (run / 'train/metrics.jsonl').read_text().splitlines()]
        assert len(metrics) == 500
        assert all(all(math.isfinite(r[k]) for k in ('loss', 'kl', 'grad_norm')) for r in metrics)
        checks = []
        for step in (100, 250, 500):
            checkpoint = torch.load(run / f'train/step-{step:06d}.ckpt',
                                    map_location='cpu', weights_only=False)
            state = checkpoint['state_dict']
            changed = [n for n,t in frozen.items() if not torch.equal(t, state[n])]
            nonzero = sum(bool(t.count_nonzero()) for n,t in state.items() if n.endswith('lora_B'))
            assert not changed and nonzero == 48, (step, changed, nonzero)
            subset = metrics[:step]
            checks.append(dict(step=step, frozen_changed=changed, nonzero_lora_b=nonzero,
                gradient_clipped_fraction=sum(r['grad_norm'] > .2 for r in subset)/step,
                max_gradient_norm=max(r['grad_norm'] for r in subset),
                mean_proxy_kl=sum(r['kl'] for r in subset)/step))
            del checkpoint, state
        (run / 'parameter_checks.json').write_text(json.dumps(checks, indent=2))
        del frozen, base
        import gc
        gc.collect()
        for step in (100, 250, 500):
            stage(f'validation-{step}')
            common = ['scripts/eval_grpo_pilot.py', '--base', args.checkpoint,
                      '--rl', str(run / f'train/step-{step:06d}.ckpt'),
                      '--rl-label', f'rl{step}', '--batch-size', '32']
            execute(common + ['--output', str(run / f'validation-{step}')],
                    f'validation-{step}.log')
            stage(f'train-probe-{step}')
            execute(common + ['--data', str(run / 'seen_train_probe32.jsonl'),
                    '--expected-count', '32', '--output', str(run / f'train-probe-{step}')],
                    f'train-probe-{step}.log')
        summaries = {str(step): json.loads(
            (run / f'validation-{step}/comparison.json').read_text())
            for step in (100, 250, 500)}
        counts = [[r['base_correct'] for r in s['comparisons']] for s in summaries.values()]
        assert counts[0] == counts[1] == counts[2], 'Base sampling changed across nodes.'
        (run / 'comparisons.json').write_text(json.dumps(summaries, indent=2))
        stage('complete')
        (run / 'COMPLETE').touch()
    except BaseException:
        (run / 'FAILED').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()

