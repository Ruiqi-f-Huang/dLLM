#!/usr/bin/env python3
"""Fixed 32-question online GRPO diagnostic; does not evaluate GSM8K Test."""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grpo_overfit import paired_bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run = Path(args.output).resolve()
    run.mkdir(parents=True, exist_ok=False)
    (run / 'pipeline.json').write_text(json.dumps(vars(args), indent=2))

    def stage(name):
        (run / 'STAGE').write_text(name+'\n')
        print(name, flush=True)

    def execute(command, label):
        stage(label)
        with (run / f'{label}.log').open('x') as log:
            subprocess.run([sys.executable, '-u'] + command, cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    try:
        sources = list(ROOT.glob('*.py'))
        for directory in ('models', 'configs', 'callbacks', 'scripts'):
            sources += [p for p in (ROOT / directory).rglob('*')
                        if p.is_file() and p.suffix in ('.py', '.yaml')]
        hashes = {}
        for p in sources:
            destination = run / 'source' / p.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, destination)
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        (run / 'source_sha256.json').write_text(json.dumps(hashes, indent=2))
        execute(['scripts/grpo_fixed32.py', 'prepare', '--audit', args.audit,
                 '--output', str(run / 'fixed')], 'prepare')
        data = str(run / 'fixed/questions.jsonl')
        common = ['scripts/grpo_fixed32.py', 'measure', '--base', args.base, '--data', data,
                  '--repeats', '16', '--batch-size', '8', '--temperatures', '.3', '.1']
        execute(common + ['--make-pairs', '--output', str(run / 'base')], 'base')
        execute(['scripts/train_mdlm_grpo.py', '--checkpoint', args.base, '--data', data,
                 '--output', str(run / 'train'), '--limit', '32', '--updates', '500',
                 '--iterations', '2', '--group-size', '8', '--microbatch', '8',
                 '--nfe', '32', '--temperature', '.3', '--seed', '42',
                 '--save-every', '500', '--save-at', '100', '250', '500',
                 '--reshuffle-cycles'], 'train')
        base_sampling = json.loads((run / 'base/sampling.json').read_text())
        base_proxy = json.loads((run / 'base/proxy.json').read_text())
        baseline = {r['sample_id']: r for r in base_proxy['per_question']}
        metrics = [json.loads(x) for x in (run / 'train/metrics.jsonl').read_text().splitlines()]
        if len(metrics) != 500 or not all(all(math.isfinite(r[k]) for k in
            ('loss', 'grad_norm', 'kl', 'policy_loss', 'kl_penalty')) for r in metrics):
            raise ValueError('Invalid or incomplete optimization metrics.')
        schedule = json.loads((run / 'train/group_schedule.json').read_text())
        counts = {sid: schedule.count(sid) for sid in set(schedule)}
        if len(counts) != 32 or max(counts.values()) - min(counts.values()) > 1:
            raise ValueError('Training schedule is not balanced.')
        # Verify the frozen EMA reference and adapter updates independently of loss.
        import torch
        import algo
        import main as entry
        checkpoint = torch.load(run / 'train/step-000100.ckpt', map_location='cpu', weights_only=False)
        base_model = entry._initialize_finetune(algo.MDLM,
            checkpoint['hyper_parameters']['config'], checkpoint['hyper_parameters']['tokenizer'])
        frozen = {n: t for n, t in base_model.state_dict().items()
                  if not n.endswith(('lora_A', 'lora_B'))}
        parameter_checks = []
        for step in (100, 250, 500):
            checkpoint = torch.load(run / f'train/step-{step:06d}.ckpt', map_location='cpu', weights_only=False)
            state = checkpoint['state_dict']
            changed = [n for n, t in frozen.items() if not torch.equal(t, state[n])]
            nonzero_b = sum(bool(t.count_nonzero()) for n, t in state.items() if n.endswith('lora_B'))
            if changed or nonzero_b != 48:
                raise ValueError(f'Parameter verification failed at {step}.')
            parameter_checks.append(dict(step=step, frozen_changed=changed, nonzero_lora_b=nonzero_b))
        (run / 'parameter_checks.json').write_text(json.dumps(parameter_checks, indent=2))
        del checkpoint, state, frozen, base_model
        summaries = []
        for step in (100, 250, 500):
            label = f'step-{step:06d}'
            execute(common + ['--checkpoint', str(run / f'train/{label}.ckpt'),
                    '--pairs', str(run / 'base/pairs.jsonl'),
                    '--output', str(run / label)], label)
            sampling = json.loads((run / label / 'sampling.json').read_text())
            proxy = json.loads((run / label / 'proxy.json').read_text())
            if {r['sample_id'] for r in proxy['per_question']} != set(baseline):
                raise ValueError('Proxy question sets changed.')
            per_question = [dict(sample_id=r['sample_id'],
                mean_logp_gap_delta=r['mean_logp_gap']-baseline[r['sample_id']]['mean_logp_gap'],
                sum_logp_gap_delta=r['sum_logp_gap']-baseline[r['sample_id']]['sum_logp_gap'])
                for r in proxy['per_question']]
            subset = metrics[:step]
            summaries.append(dict(step=step, sampling=sampling,
                paired_sampling={t: paired_bootstrap(base_sampling[t], sampling[t]) for t in sampling},
                proxy_gap_deltas=per_question,
                mean_proxy_gap_delta=sum(r['mean_logp_gap_delta'] for r in per_question)/32,
                gradient_clipped_fraction=sum(r['grad_norm'] > .2 for r in subset)/step,
                mean_policy_loss=sum(r['policy_loss'] for r in subset)/step,
                mean_kl_penalty=sum(r['kl_penalty'] for r in subset)/step))
            (run / 'comparison.json').write_text(json.dumps(dict(base=base_sampling,
                stages=summaries, training_group_counts=counts,
                limitations=['Selected mixed-reward training questions, not generalization.',
                    'One training seed; repeated samples are clustered by question.',
                    'Binary mean reward and success@1 estimate are the same statistic.',
                    'Empirical pass@16 may saturate; fixed-pair gaps are proxy diagnostics.',
                    'Full-mask proxy scores are untempered; rollout uses T=.3.',
                    'No validation, Test, LR or KL sweep is run by this pipeline.']), indent=2))
        stage('complete')
        (run / 'COMPLETE').touch()
    except BaseException:
        (run / 'FAILED').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
