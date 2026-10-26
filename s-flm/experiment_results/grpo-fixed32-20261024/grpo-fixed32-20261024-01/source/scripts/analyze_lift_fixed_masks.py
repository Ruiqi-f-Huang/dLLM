"""Question-cluster paired bootstrap of existing fixed-mask diagnostic arrays."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_lift_fixed_masks import REGIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    root = args.root
    assert (root / 'COMPLETE').exists(), 'Diagnostic is not complete'
    manifest = json.loads((root / 'manifest.json').read_text())
    assert set(manifest['splits']['train256']['sample_ids']).isdisjoint(
        manifest['splits']['validation632']['sample_ids'])
    comparisons = [('c100','base'),('c200','base'),('c100','a100'),('c200','a200'),
                   ('c100','b100'),('c200','b200'),('c200','c100'),
                   ('a200','a100'),('b200','b100')]
    rows = []
    for split in ('train256','validation632'):
        n = len(manifest['splits'][split]['sample_ids'])
        rng = np.random.default_rng(20261023)
        weights = rng.multinomial(n, np.full(n,1/n), size=args.bootstrap).astype(np.float64)
        for q in (.25,.5,.75,.9,1.):
            seeds = [17] if q == 1 else [17,29]
            models = {}
            for label in ('base','a100','a200','b100','b200','c100','c200'):
                arrays = [np.load(root / f'{label}-{split}-q{q:g}-seed{s}.npz')['stats'] for s in seeds]
                a = np.stack(arrays).sum(0)
                assert a.shape == (n,len(REGIONS),3) and np.isfinite(a).all()
                assert (a[:,:,1] <= a[:,:,2]).all()
                models[label] = a
                if label != 'base':
                    np.testing.assert_array_equal(a[:,:,2], models['base'][:,:,2])
            tokens = models['base'][:,:,2]
            denominator = weights @ tokens
            for candidate, reference in comparisons:
                c, r = models[candidate], models[reference]
                diff = c - r
                bootstrap_ce = (weights @ diff[:,:,0]) / denominator
                bootstrap_acc = (weights @ diff[:,:,1]) / denominator
                for i,region in enumerate(REGIONS):
                    count = tokens[:,i].sum()
                    assert count > 0
                    valid = tokens[:,i] > 0
                    question_delta = diff[valid,i,0] / tokens[valid,i]
                    rows.append({'split':split,'rate':q,'mask_seeds':seeds,
                        'candidate':candidate,'reference':reference,'region':region,
                        'tokens':int(count),'scored_questions':int(valid.sum()),
                        'candidate_ce':float(c[:,i,0].sum()/count),
                        'reference_ce':float(r[:,i,0].sum()/count),
                        'delta_ce':float(diff[:,i,0].sum()/count),
                        'ci95_delta_ce':np.quantile(bootstrap_ce[:,i],[.025,.975]).tolist(),
                        'candidate_accuracy':float(c[:,i,1].sum()/count),
                        'reference_accuracy':float(r[:,i,1].sum()/count),
                        'delta_accuracy_pp':float(100*diff[:,i,1].sum()/count),
                        'delta_correct_tokens':int(diff[:,i,1].sum()),
                        'ci95_delta_accuracy_pp':(100*np.quantile(bootstrap_acc[:,i],[.025,.975])).tolist(),
                        'questions_ce_improved':int((question_delta < -1e-8).sum()),
                        'questions_ce_worsened':int((question_delta > 1e-8).sum())})
    result = {'bootstrap':args.bootstrap,'unit':'question; both partial-mask seeds kept within each resampled question',
              'confidence_intervals':'percentile 95%; exploratory, no multiplicity correction',
              'delta_direction':'negative CE / positive accuracy is improvement', 'rows':rows}
    (root/'paired_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in rows:
        if row['split']=='validation632' and row['region']=='critical' and row['rate']>=.75:
            print(json.dumps(row))


if __name__=='__main__':
    main()
