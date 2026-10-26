"""Append aggregate validation results to the already synced C training run."""
import argparse
import json
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--run-path', default='ruiqi-f-huang-nus/s-flm/zog10k73')
    args = parser.parse_args()
    entity, project, run_id = args.run_path.split('/')
    result = json.loads((args.root / 'comparison.json').read_text())
    rows = {(r['label'], r['seed']): r for r in result['rows']}
    api = wandb.Api(timeout=60)
    existing = api.run(args.run_path)
    if existing.summary.get('lift_validation_uploaded_version') == 1:
        print('Evaluation already uploaded:', existing.url)
        return
    payload = {
        'protocol': result['protocol'],
        'method': 'MDLM code/EOS-region LIFT-3; docstring/PAD retain random masking',
        'means': {k: result['means'][k] for k in ('base', 'c100', 'c200')},
        'rows': [r for r in result['rows'] if r['label'] in ('base', 'c100', 'c200')],
        'training': result['training']['c'],
        'limitations': result['limitations'],
    }
    path = args.root / 'wandb_c_evaluation.json'
    path.write_text(json.dumps(payload, indent=2) + '\n')
    with wandb.init(entity=entity, project=project, id=run_id,
                    resume='must', mode='online') as run:
        run.define_metric('evaluation/checkpoint_step')
        metric_names = ['mean_accuracy', 'base_mean_accuracy', 'delta_mean_correct_vs_base']
        for seed in (1, 2):
            metric_names += [f'accuracy_seed{seed}', f'correct_seed{seed}',
                             f'base_accuracy_seed{seed}', f'gained_vs_base_seed{seed}',
                             f'lost_vs_base_seed{seed}', f'paired_p_vs_base_seed{seed}']
        for name in metric_names:
            run.define_metric(f'evaluation/{name}', step_metric='evaluation/checkpoint_step')
        table = wandb.Table(columns=['model', 'checkpoint', 'seed', 'correct', 'total',
                                     'accuracy', 'raw_parseable', 'gained_vs_base',
                                     'lost_vs_base', 'paired_p_vs_base'])
        for label in ('base', 'c100', 'c200'):
            for seed in (1, 2):
                row = rows[label, seed]
                table.add_data(label, 0 if label == 'base' else int(label[1:]), seed,
                               row['correct'], row['total'], row['accuracy'],
                               row['raw_parseable'], row['vs_base']['gained'],
                               row['vs_base']['lost'], row['vs_base']['paired_exact_p'])
        for step in (100, 200):
            label = f'c{step}'
            values = {'evaluation/checkpoint_step': step,
                      'evaluation/mean_accuracy': result['means'][label]['mean_accuracy'],
                      'evaluation/base_mean_accuracy': result['means']['base']['mean_accuracy'],
                      'evaluation/delta_mean_correct_vs_base': sum(
                          rows[label, s]['correct'] - rows['base', s]['correct'] for s in (1, 2)) / 2}
            for seed in (1, 2):
                row, base = rows[label, seed], rows['base', seed]
                values.update({f'evaluation/accuracy_seed{seed}': row['accuracy'],
                               f'evaluation/correct_seed{seed}': row['correct'],
                               f'evaluation/base_accuracy_seed{seed}': base['accuracy'],
                               f'evaluation/gained_vs_base_seed{seed}': row['vs_base']['gained'],
                               f'evaluation/lost_vs_base_seed{seed}': row['vs_base']['lost'],
                               f'evaluation/paired_p_vs_base_seed{seed}': row['vs_base']['paired_exact_p']})
            if step == 200:
                values['evaluation/checkpoint_results'] = table
            # Resume appends history rows; the custom axis is checkpoint_step.
            # Do not attempt to backfill W&B's already completed training steps.
            run.log(values)
        run.summary['evaluation/protocol'] = result['protocol']
        run.summary['evaluation/conclusion'] = 'No robust improvement; C200 averages +0.5 question vs Base across two generation seeds.'
        artifact = wandb.Artifact(f'lift-c-validation-{run_id}', type='evaluation',
                                  description='Aggregate fixed-validation C100/C200 and Base results; no raw questions or weights.')
        artifact.add_file(str(path))
        run.log_artifact(artifact).wait()
    verified = wandb.Api(timeout=60).run(args.run_path)
    assert verified.summary['evaluation/correct_seed1'] == 180
    assert verified.summary['evaluation/correct_seed2'] == 163
    assert verified.summary['monitor/successful_updates_session'] == 200
    verified.summary['lift_validation_uploaded_version'] = 1
    verified.summary.update()
    (args.root / 'wandb_c_upload.json').write_text(json.dumps({
        'run_path': args.run_path, 'url': verified.url,
        'training_successful_updates': 200,
        'validation_checkpoints': [100, 200], 'generation_seeds': [1, 2],
        'aggregate_artifact': f'{entity}/{project}/lift-c-validation-{run_id}',
    }, indent=2) + '\n')
    print('Verified:', verified.url)


if __name__ == '__main__':
    main()
