#!/usr/bin/env python3
"""Four controlled fixed-program fits; no sampling or online RL parameter sweep."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True)
    p.add_argument('--fixed-run',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=200)
    args=p.parse_args()
    run=Path(args.output).resolve();run.mkdir(parents=True,exist_ok=False)
    (run/'pipeline.json').write_text(json.dumps(vars(args),indent=2))
    fixed=Path(args.fixed_run).resolve()
    try:
        for path in [fixed/'fixed/questions.jsonl',fixed/'base/pairs.jsonl']:
            shutil.copy2(path,run/path.name)
        for name in ['grpo_controlled.py','diffu_grpo.py','models/dit.py',
                     'scripts/diagnose_grpo_fixed.py','scripts/run_grpo_controlled.py']:
            dst=run/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,dst)
        summaries={}
        for precision in ['bfloat16','float32']:
            for lr in [3e-6,3e-5]:
                name=f'{precision}-lr{lr:g}'
                (run/'STAGE').write_text(name+'\n')
                print(name,flush=True)
                with (run/f'{name}.log').open('x') as log:
                    subprocess.run([sys.executable,'-u','scripts/diagnose_grpo_fixed.py',
                        '--base',args.base,'--questions',str(run/'questions.jsonl'),
                        '--pairs',str(run/'pairs.jsonl'),'--output',str(run/name),
                        '--precision',precision,'--lr',str(lr),'--steps',str(args.steps)],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                summaries[name]=json.loads((run/name/'summary.json').read_text())
                (run/'comparison.json').write_text(json.dumps(summaries,indent=2))
        (run/'STAGE').write_text('complete\n');(run/'COMPLETE').touch()
    except BaseException:
        (run/'FAILED').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
