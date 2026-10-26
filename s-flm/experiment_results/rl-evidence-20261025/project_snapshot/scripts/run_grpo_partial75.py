#!/usr/bin/env python3
"""One partial75 training arm, paired generation and fixed-state diagnostics."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grpo_overfit import paired_bootstrap

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['base','online-run','trajectory-run','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();run=Path(a.output).resolve();run.mkdir(parents=True,exist_ok=False)
    online=Path(a.online_run)
    (run/'pipeline.json').write_text(json.dumps(vars(a),indent=2))
    def execute(command,label):
        (run/'STAGE').write_text(label+'\n');print(label,flush=True)
        with (run/(label+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-u']+command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        if not (online/'COMPLETE').exists():raise ValueError('Online control incomplete.')
        for name in ['questions.jsonl','pairs.jsonl']:shutil.copy2(online/name,run/name)
        for name in ['diffu_grpo.py','grpo_monitor.py','scripts/train_mdlm_grpo.py','scripts/run_grpo_partial75.py','scripts/diagnose_grpo_trajectory.py']:
            dest=run/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        execute(['scripts/train_mdlm_grpo.py','--checkpoint',a.base,'--data',str(run/'questions.jsonl'),
            '--output',str(run/'train'),'--precision','float32','--monitor','--completion-mask','.75',
            '--limit','32','--updates','250','--iterations','2','--group-size','8','--microbatch','8',
            '--nfe','32','--temperature','.3','--lr','3e-5','--seed','42','--save-every','250',
            '--save-at','100','250','--reshuffle-cycles'],'train')
        checkpoint=run/'train/step-000250.ckpt'
        execute(['scripts/grpo_fixed32.py','measure','--base',a.base,'--checkpoint',str(checkpoint),
            '--data',str(run/'questions.jsonl'),'--pairs',str(run/'pairs.jsonl'),
            '--precision','float32','--execution-details','--repeats','16','--batch-size','8',
            '--temperatures','.3','.1','--output',str(run/'evaluation')],'evaluation')
        sampling=json.loads((run/'evaluation/sampling.json').read_text())
        comparisons={}
        for label,path in [('base',online/'base/sampling.json'),('fullmask250',online/'lr3e-05/step-000250/sampling.json')]:
            control=json.loads(path.read_text())
            comparisons[label]={t:paired_bootstrap(control[t],sampling[t]) for t in sampling}
        (run/'sampling_comparison.json').write_text(json.dumps(comparisons,indent=2))
        execute(['scripts/diagnose_grpo_trajectory.py','--base',a.base,'--online-run',a.online_run,
            '--trajectory-run',a.trajectory_run,'--candidate-checkpoint',str(checkpoint),
            '--output',str(run/'trajectory')],'trajectory')
        (run/'STAGE').write_text('complete\n');(run/'COMPLETE').touch()
    except BaseException:
        (run/'FAILED').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
