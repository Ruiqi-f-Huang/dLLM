"""Paired validation comparison for the 711-step semantic weighting experiment."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sandbox_gsm8k import evaluate_samples, _extract_code

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
root = args.run

def load(path):
    data = json.loads(path.read_text())
    records = data['records']
    flags = [bool(evaluate_samples(r['response'], r['response_ground_truth'], 5.)) for r in records]
    assert sum(flags) == data['num_correct'], (path, sum(flags), data['num_correct'])
    return records, flags


def paired(a, b):
    ar, af = a
    br, bf = b
    assert len(ar) == len(br) == 632
    assert all((x['prompt'], x['response_ground_truth']) == (y['prompt'], y['response_ground_truth']) for x, y in zip(ar, br))
    return dict(reference_correct=sum(af), candidate_correct=sum(bf),
                kept=sum(x and y for x,y in zip(af,bf)),
                lost=sum(x and not y for x,y in zip(af,bf)),
                gained=sum(not x and y for x,y in zip(af,bf)),
                identical_response=sum(x['response']==y['response'] for x,y in zip(ar,br)))


def formatting(records):
    parseable = 0
    for record in records:
        try:
            compile(_extract_code(record['response']), '<generated>', 'exec')
            parseable += 1
        except (SyntaxError, ValueError, TypeError):
            pass
    return dict(parseable=parseable, unclosed_quotes=sum(r['response'].count('"""') % 2 or r['response'].count("'''") % 2 for r in records))

old = {step: load(root/'historical-control'/f'step-{step}-ema-steps32-seed1/results.json') for step in [0,100,200,711]}
new = {step: load(root/'eval'/f'step-{step}-ema-steps32-seed1/results.json') for step in [0,100,200,711]}
control = load(root/'retained-control/step-200-ema-steps32-seed1/results.json')
summary = dict(protocol='632 validation, EMA, ancestral32, seed1, length512, eval_batch32',
               hardware_sanity={'base':paired(old[0],new[0]),'control200':paired(old[200],control)},
               weighted_milestones=[], historical_vs_weighted=[])
for step in [0,100,200,711]:
    summary['weighted_milestones'].append(dict(step=step, **paired(new[0],new[step]), **formatting(new[step][0])))
    summary['historical_vs_weighted'].append(dict(step=step, **paired(old[step],new[step])))
(root/'paired_comparison.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
