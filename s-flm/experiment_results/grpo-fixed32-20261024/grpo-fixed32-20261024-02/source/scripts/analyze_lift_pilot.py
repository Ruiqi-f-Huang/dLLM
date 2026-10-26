"""Paired low-temperature validation and frozen-weight audit of LIFT pilot."""
import argparse
import ast
from collections import Counter
import json
import math
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from omegaconf import OmegaConf
from classify_base_lowtemp_errors import audit
from sandbox_gsm8k import _extract_code


def compare(reference, candidate):
    gains = sum(not a and b for a, b in zip(reference['flags'], candidate['flags']))
    losses = sum(a and not b for a, b in zip(reference['flags'], candidate['flags']))
    n = gains + losses
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2**n) if n else 1.
    return {'gained': gains, 'lost': losses, 'net': gains - losses,
            'paired_exact_p': p,
            'changed_outputs': sum(a['response'] != b['response'] for a, b in zip(reference['records'], candidate['records'])),
            'changed_ast_excluding_docstrings': sum(a != b for a, b in zip(reference['asts'], candidate['asts'])
                                                     if a is not None and b is not None)}


def normalized_ast(response):
    try:
        code = _extract_code(response); compile(code, '<sample>', 'exec')
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
        return ast.dump(tree, include_attributes=False)
    except (SyntaxError, ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('run', type=Path)
    run = parser.parse_args().run
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    labels = ['base', 'a100', 'a200', 'b100', 'b200', 'c100', 'c200']
    models, rows = {}, []
    for seed in (1, 2):
        for label in labels:
            directory = run / 'evaluation' / f'{label}-seed{seed}'
            cfg = OmegaConf.load(directory / '.hydra/config.yaml')
            assert cfg.seed == seed and cfg.sampler.steps == 32 and cfg.sampler.temperature == .1
            assert cfg.sampler.predictor == 'ancestral' and not cfg.eval.disable_ema
            assert cfg.model.length == 512 and cfg.loader.eval_batch_size == 32
            assert cfg.lora.enabled == (label != 'base')
            data = json.loads((directory / 'results.json').read_text())
            records = data['records']; assert len(records) == 632
            audits = [audit(record, i) for i, record in enumerate(records)]
            flags = [r['status'] == 'correct' for r in audits]
            assert sum(flags) == data['num_correct']
            (directory / 'execution_audit.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in audits))
            models[label, seed] = {'records': records, 'flags': flags,
                                   'asts': [normalized_ast(r['response']) for r in records]}
            reference = models['base', seed]
            assert [(r['prompt'], r['response_ground_truth']) for r in records] == [
                (r['prompt'], r['response_ground_truth']) for r in reference['records']]
            row = {'label': label, 'seed': seed, 'correct': sum(flags), 'total': 632,
                   'accuracy': sum(flags) / 632,
                   'raw_parseable': sum(r['raw_parseable'] for r in audits),
                   'recovered_status_counts': dict(Counter(r['status'] for r in audits)),
                   'strict_status_counts': dict(Counter(r['status'] if r['raw_parseable'] else 'parse_error' for r in audits)),
                   'undefined_variable_errors': sum(r.get('error_type') in ('NameError', 'UnboundLocalError') for r in audits),
                   'unclosed_quotes': sum(bool(r['response'].count('"""') % 2 or r['response'].count("'''") % 2) for r in records),
                   'vs_base': compare(reference, models[label, seed])}
            if label != 'base':
                step = label[1:]
                row['vs_a'] = compare(models[f'a{step}', seed], models[label, seed])
                if label.startswith('c'):
                    row['vs_b'] = compare(models[f'b{step}', seed], models[label, seed])
            rows.append(row)
    # Check completed updates and imported/frozen Base parameters independently
    # of generation metrics. This creates a CPU reference, not a training run.
    import torch
    import main as entry
    import algo
    import dataloader
    training = {}
    cfg = OmegaConf.load(run / 'train-a/.hydra/config.yaml')
    tokenizer = dataloader.get_tokenizer(cfg)
    reference = entry._initialize_finetune(algo.MDLM, cfg, tokenizer)
    frozen = {n: p.detach() for n, p in reference.named_parameters() if not p.requires_grad}
    for arm in 'abc':
        summary = json.loads((run / f'train-{arm}/metrics/version_0/summary.json').read_text())
        assert summary['final_step'] == summary['successful_updates_session'] == 200
        assert summary['amp_skipped_updates_session'] == summary['nonfinite_gradient_updates_session'] == 0
        checks = {}
        for step in (100, 200):
            ckpt = torch.load(run / f'train-{arm}/checkpoints/step-{step:06d}.ckpt', map_location='cpu', weights_only=False)
            state = ckpt['state_dict']
            assert all(torch.equal(state[n], p) for n, p in frozen.items())
            checks[step] = {'frozen_parameters_unchanged': True,
                            'nonzero_lora_b_tensors': sum(bool(p.abs().sum()) for n, p in state.items() if n.endswith('lora_B')),
                            'ema_updates': ckpt['ema']['num_updates']}
            del ckpt, state
        training[arm] = {'summary': summary, 'checkpoints': checks}
    assert len({v['summary']['examples_seen_session'] for v in training.values()}) == 1
    means = {label: {'correct_by_seed': [sum(models[label, seed]['flags']) for seed in (1, 2)],
                     'mean_accuracy': sum(sum(models[label, seed]['flags']) for seed in (1, 2)) / (2 * 632)}
             for label in labels}
    result = {'protocol': {'split': 'validation', 'questions': 632, 'seeds': [1, 2],
                           'temperature': .1, 'nfe': 32, 'sampler': 'ancestral', 'weights': 'EMA',
                           'hardware': 'single A800', 'training_seed': 1},
              'training': training, 'means': means, 'rows': rows,
              'limitations': ['One training seed; generation seeds repeat the same questions.',
                              'Paired p values are exploratory and not multiplicity corrected.',
                              'LIFT is restricted to code/EOS; this is not full-region original LIFT.']}
    (run / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    with (run / 'paired_by_question.jsonl').open('w') as out:
        for i, record in enumerate(models['base', 1]['records']):
            out.write(json.dumps({'index': i, 'question': record['prompt'],
                                  'correct': {f'{label}-seed{seed}': models[label, seed]['flags'][i]
                                              for label in labels for seed in (1, 2)}}) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
