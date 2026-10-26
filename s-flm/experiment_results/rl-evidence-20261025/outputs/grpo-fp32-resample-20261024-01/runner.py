#!/usr/bin/env python3
"""Independent paired resampling of the preselected Base / high-LR final node."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grpo_overfit import summarize_samples, paired_bootstrap


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-run',required=True)
    p.add_argument('--base',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--eval-seed',type=int,default=20261025)
    args=p.parse_args()
    original=Path(args.original_run).resolve()
    if not (original/'COMPLETE').exists():raise ValueError('Original experiment incomplete.')
    model=original/'lr3e-05/train/step-000250.ckpt'
    data=original/'questions.jsonl';pairs=original/'pairs.jsonl'
    for source in [model,data,pairs]:
        if not source.is_file():raise FileNotFoundError(source)
    run=Path(args.output).resolve();run.mkdir(parents=True,exist_ok=False)
    (run/'run.json').write_text(json.dumps(vars(args),indent=2))
    (run/'runner.py').write_text(Path(__file__).read_text())
    try:
        common=['scripts/grpo_fixed32.py','measure','--base',args.base,'--data',str(data),
                '--pairs',str(pairs),'--precision','float32','--execution-details',
                '--repeats','16','--batch-size','8','--temperatures','.3','.1',
                '--eval-seed',str(args.eval_seed)]
        for label in ['base','rl250']:
            (run/'STAGE').write_text(label+'\n')
            command=common+['--output',str(run/label)]
            if label=='rl250':command += ['--checkpoint',str(model)]
            with (run/f'{label}.log').open('x') as log:
                subprocess.run([sys.executable,'-u']+command,cwd=ROOT,
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        result=dict(protocol=dict(selected_node='FP32 LR=3e-5 step250',
                    fresh_eval_seed=args.eval_seed,repeats_per_question_per_batch=16,
                    combined_repeats=32,n_questions=32,temperatures=[.3,.1]),
                    fresh={},combined={},original={},
                    limitations=['Same selected training questions and one training seed.',
                                 'Original and fresh batch reported separately; no node reselection.',
                                 'Bootstrap clusters repeated samples by question.'])
        for t in ['0.3','0.1']:
            aggregates={}
            fresh={}
            for label,previous in [('base',original/'base'),
                                   ('rl250',original/'lr3e-05/step-000250')]:
                old_run=json.loads((previous/'run.json').read_text())
                if old_run['eval_seed']==args.eval_seed:raise ValueError('Not an independent seed.')
                old=read(previous/f'samples-t{t}.jsonl')
                new=read(run/label/f'samples-t{t}.jsonl')
                if {x['seed'] for x in old}&{x['seed'] for x in new}:
                    raise ValueError('Sampling seed collision.')
                fresh[label]=summarize_samples(new,16)
                aggregates[label]=summarize_samples(old+[dict(x,repeat=x['repeat']+16) for x in new],32)
                result['original'].setdefault(t,{})[label]=summarize_samples(old,16)
            result['fresh'][t]=dict(**fresh,paired=paired_bootstrap(fresh['base'],fresh['rl250']))
            result['combined'][t]=dict(**aggregates,paired=paired_bootstrap(aggregates['base'],aggregates['rl250']))
        (run/'comparison.json').write_text(json.dumps(result,indent=2))
        (run/'STAGE').write_text('complete\n');(run/'COMPLETE').touch()
    except BaseException:
        (run/'FAILED').write_text(traceback.format_exc());raise


if __name__=='__main__':main()

