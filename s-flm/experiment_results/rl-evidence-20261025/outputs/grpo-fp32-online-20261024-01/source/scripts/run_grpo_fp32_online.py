#!/usr/bin/env python3
"""Monitored FP32 online LR comparison on the frozen mixed-reward 32 questions."""
import argparse
import gc
import hashlib
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
    p.add_argument('--base',required=True)
    p.add_argument('--fixed-run',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=250)
    args=p.parse_args()
    if args.steps < 2: p.error('Need at least two updates.')
    run=Path(args.output).resolve();run.mkdir(parents=True,exist_ok=False)
    fixed=Path(args.fixed_run).resolve()
    if not (fixed/'COMPLETE').exists():raise ValueError('Source fixed32 run is incomplete.')
    (run/'pipeline.json').write_text(json.dumps(vars(args),indent=2))
    def execute(command,label):
        (run/'STAGE').write_text(label+'\n');print(label,flush=True)
        with (run/f'{label}.log').open('x') as log:
            subprocess.run([sys.executable,'-u']+command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        for source in [fixed/'fixed/questions.jsonl',fixed/'base/pairs.jsonl']:
            shutil.copy2(source,run/source.name)
        hashes={}
        paths=list(ROOT.glob('*.py'))
        for directory in ['models','configs','callbacks','scripts']:
            paths += [p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml')]
        for path in paths:
            destination=run/'source'/path.relative_to(ROOT)
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
            hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        (run/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
        measurement=['scripts/grpo_fixed32.py','measure','--base',args.base,
            '--data',str(run/'questions.jsonl'),'--pairs',str(run/'pairs.jsonl'),
            '--precision','float32','--execution-details','--repeats','16','--batch-size','8',
            '--temperatures','.3','.1']
        # The Base must be remeasured in FP32, not reused from BF16 results.
        execute(measurement+['--output',str(run/'base')],'base')
        baseline=json.loads((run/'base/sampling.json').read_text())
        base_proxy=json.loads((run/'base/proxy.json').read_text())
        stages=[s for s in [100,args.steps] if s<=args.steps]
        stages=sorted(set(stages))
        comparisons={}
        for lr in [3e-6,3e-5]:
            name=f'lr{lr:g}'
            execute(['scripts/train_mdlm_grpo.py','--checkpoint',args.base,
                '--data',str(run/'questions.jsonl'),'--output',str(run/name/'train'),
                '--precision','float32','--monitor','--diagnostic-pairs',str(run/'pairs.jsonl'),
                '--limit','32','--updates',str(args.steps),'--iterations','2','--group-size','8',
                '--microbatch','8','--nfe','32','--temperature','.3','--lr',str(lr),
                '--seed','42','--save-every',str(args.steps),'--save-at',*[str(s) for s in stages],
                '--reshuffle-cycles'],name+'-train')
            import torch
            import algo
            import main as entry
            checkpoint=torch.load(run/name/f'train/step-{stages[0]:06d}.ckpt',map_location='cpu',weights_only=False)
            model=entry._initialize_finetune(algo.MDLM,checkpoint['hyper_parameters']['config'],
                                          checkpoint['hyper_parameters']['tokenizer'])
            frozen={n:p for n,p in model.state_dict().items() if not n.endswith(('lora_A','lora_B'))}
            checks=[]
            for step in stages:
                checkpoint=torch.load(run/name/f'train/step-{step:06d}.ckpt',map_location='cpu',weights_only=False)
                state=checkpoint['state_dict']
                changed=[n for n,p in frozen.items() if not torch.equal(p,state[n])]
                nonzero=sum(bool(p.count_nonzero()) for n,p in state.items() if n.endswith('lora_B'))
                if changed or nonzero!=48:raise ValueError('Parameter verification failed.')
                if 'diagnostic_state' not in checkpoint:raise ValueError('RNG/schedule state missing.')
                checks.append(dict(step=step,frozen_changed=changed,nonzero_lora_b=nonzero))
            (run/name/'parameter_checks.json').write_text(json.dumps(checks,indent=2))
            del checkpoint,state,frozen,model;gc.collect()
            results={}
            for step in stages:
                label=f'{name}-step{step}'
                destination=run/name/f'step-{step:06d}'
                execute(measurement+['--checkpoint',str(run/name/f'train/step-{step:06d}.ckpt'),
                                    '--output',str(destination)],label)
                sampling=json.loads((destination/'sampling.json').read_text())
                proxy=json.loads((destination/'proxy.json').read_text())
                proxy_base={r['sample_id']:r for r in base_proxy['per_question']}
                deltas=[dict(sample_id=r['sample_id'],
                    success_mean_logp_delta=r['success']['mean_logp']-proxy_base[r['sample_id']]['success']['mean_logp'],
                    failure_mean_logp_delta=r['failure']['mean_logp']-proxy_base[r['sample_id']]['failure']['mean_logp'],
                    gap_delta=r['mean_logp_gap']-proxy_base[r['sample_id']]['mean_logp_gap']) for r in proxy['per_question']]
                results[str(step)]=dict(sampling=sampling,
                    paired_sampling={t:paired_bootstrap(baseline[t],sampling[t]) for t in sampling},proxy_deltas=deltas)
            comparisons[name]=results
            (run/'comparison.json').write_text(json.dumps(dict(base=baseline,conditions=comparisons,
                limitations=['One training seed; selected training questions, not generalization.',
                    'Checkpoint contains RNG/cache state; full resume is not implemented or validated.',
                    'No validation/Test or automatic extra sampling is run.']),indent=2))
        (run/'STAGE').write_text('complete\n');(run/'COMPLETE').touch()
    except BaseException:
        (run/'FAILED').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
