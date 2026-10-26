#!/usr/bin/env python3
"""Evaluate positive-only fixed8 adapter: generation and fixed real trajectories."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['base','online-run','positive-run','trajectory-run','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();run=Path(a.output).resolve();run.mkdir(parents=True,exist_ok=False)
    (run/'run.json').write_text(json.dumps(vars(a),indent=2));shutil.copy2(__file__,run/'runner.py')
    def execute(command,label):
        (run/'STAGE').write_text(label+'\n');print(label,flush=True)
        with (run/(label+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-u']+command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        import torch
        import lightning as L
        from grpo_fixed32 import load_model
        from grpo_overfit import paired_bootstrap
        from grpo_partial_mask import paired_summary
        source=Path(a.positive_run);online=Path(a.online_run)
        for directory in [source,online,Path(a.trajectory_run)]:
            if not (directory/'COMPLETE').exists():raise ValueError('Source incomplete: '+str(directory))
        artifact=torch.load(source/'final_adapters.pt',map_location='cpu',weights_only=False)
        if artifact['args']['steps']!=200:raise ValueError('Expected positive step200 artifact.')
        train_ids={x['sample_id'] for x in json.loads((source/'fixed_batch.json').read_text())['records']}
        if len(train_ids)!=8:raise ValueError('Expected fixed8 subset.')
        (run/'training_ids.json').write_text(json.dumps(sorted(train_ids),indent=2))
        (run/'STAGE').write_text('restore-adapter\n');print('restore-adapter',flush=True)
        model,tok=load_model(a.base,None);state=model.state_dict()
        expected={n for n,p in model.named_parameters() if p.requires_grad}
        if expected!=set(artifact['adapters']):raise ValueError('Adapter keys differ.')
        for name,value in artifact['adapters'].items():
            if state[name].shape!=value.shape or not torch.isfinite(value).all():raise ValueError('Invalid adapter tensor.')
            state[name]=value.to(state[name].device)
        model.load_state_dict(state,strict=True);model.config.backbone_compute_dtype='float32'
        checkpoint=run/'positive-step200.ckpt'
        torch.save(dict(state_dict=model.state_dict(),hyper_parameters=dict(model.hparams),
            **{'pytorch-lightning_version':L.__version__},global_step=200,grpo_args=artifact['args']),checkpoint)
        # Verify conversion by reloading through the same loader used by both evaluators.
        expected_values={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in expected}
        del model,state;gc.collect();torch.cuda.empty_cache()
        verify,_=load_model(a.base,str(checkpoint))
        if any(not torch.equal(p.detach().cpu(),expected_values[n]) for n,p in verify.named_parameters() if n in expected):raise ValueError('Checkpoint conversion changed adapters.')
        del verify,artifact,expected_values;gc.collect();torch.cuda.empty_cache()
        execute(['scripts/grpo_fixed32.py','measure','--base',a.base,'--checkpoint',str(checkpoint),
            '--data',str(online/'questions.jsonl'),'--pairs',str(online/'pairs.jsonl'),
            '--precision','float32','--execution-details','--repeats','16','--batch-size','8',
            '--temperatures','.3','.1','--output',str(run/'generation')],'generation')
        base=json.loads((online/'base/sampling.json').read_text());current=json.loads((run/'generation/sampling.json').read_text())
        comparisons={}
        for group in ['all32','trained8','other24']:
            def select(rows):return [x for x in rows if group=='all32' or ((x['sample_id'] in train_ids)==(group=='trained8'))]
            comparisons[group]={}
            for t in current:
                b=select(base[t]['per_question']);c=select(current[t]['per_question'])
                comparisons[group][t]=dict(questions=len(c),base_correct=sum(x['correct'] for x in b),current_correct=sum(x['correct'] for x in c),draws=sum(x['draws'] for x in c),**paired_bootstrap(dict(per_question=b),dict(per_question=c)))
        (run/'generation_comparison.json').write_text(json.dumps(comparisons,indent=2))
        execute(['scripts/diagnose_grpo_trajectory.py','--base',a.base,'--online-run',a.online_run,
            '--trajectory-run',a.trajectory_run,'--candidate-checkpoint',str(checkpoint),
            '--output',str(run/'trajectory')],'trajectory')
        macro=json.loads((run/'trajectory/per_question.json').read_text());out=[]
        keys=sorted({(x['reward'],x['step'],x['region'],x['temperature']) for x in macro})
        for group in ['trained8','other24']:
            for reward,step,region,temp in keys:
                def select(label):return [x for x in macro if x['model']==label and (x['sample_id'] in train_ids)==(group=='trained8') and (x['reward'],x['step'],x['region'],x['temperature'])==(reward,step,region,temp)]
                out.append(dict(group=group,reward=reward,step=step,region=region,temperature=temp,**paired_summary(select('base'),select('rl250'))))
        (run/'trajectory_subgroups.json').write_text(json.dumps(out,indent=2))
        (run/'limitations.json').write_text(json.dumps(['One seed; chosen 32 training questions, no validation/test claims.','Other24 were not used in this fixed8 fit, but were selected in earlier experiments.','16 draws/question and two fixed Base trajectories/question; limited statistical power.','Trajectory label rl250 identifies the positive step200 candidate here; see run.json.'],indent=2))
        (run/'COMPLETE').touch();(run/'STAGE').write_text('complete\n')
    except BaseException:
        (run/'FAILED').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
