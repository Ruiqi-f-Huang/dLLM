#!/usr/bin/env python3
"""Read-only fixed-pair partial75 scores and signed proxy gradient alignment."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['base','online-run','partial-run','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args()
    import torch
    from grpo_fixed32 import load_model,read_jsonl
    from diffu_grpo import completion_mask,partial_completion_mask
    from grpo_overfit import stable_seed
    from grpo_partial_mask import paired_summary
    from grpo_controlled import flatten_gradients
    online=Path(a.online_run);partial=Path(a.partial_run);run=Path(a.output)
    if not (online/'COMPLETE').exists() or not (partial/'COMPLETE').exists():raise ValueError('Sources incomplete.')
    run.mkdir(parents=True,exist_ok=False)
    (run/'run.json').write_text(json.dumps(vars(a),indent=2));(run/'runner.py').write_text(Path(__file__).read_text())
    try:
        pairs=read_jsonl(online/'pairs.jsonl');questions={x['sample_id']:x for x in read_jsonl(online/'questions.jsonl')}
        if len(pairs)!=64 or len({(x['sample_id'],x['reward']) for x in pairs})!=64 or len(questions)!=32:raise ValueError('Pair coverage mismatch.')
        plan=None;scores=[];vectors={};initial=None;delta=None;names=None
        checkpoint=partial/'train/step-000250.ckpt'
        for label,path in [('base',None),('partial250',str(checkpoint))]:
            (run/'STAGE').write_text(label+'\n');print(label,flush=True)
            model,tok=load_model(a.base,path);model.config.backbone_compute_dtype='float32';model.eval()
            if model.config.algo.time_conditioning:raise ValueError('Expected time-unconditioned model.')
            params=[(n,p) for n,p in model.named_parameters() if n.endswith(('lora_A','lora_B'))]
            if not params:raise ValueError('Missing adapters.')
            for n,p in model.named_parameters():p.requires_grad_(n.endswith(('lora_A','lora_B')))
            if initial is None:
                names=[n for n,p in params];initial=torch.cat([p.detach().float().cpu().flatten() for n,p in params])
            else:
                if names!=[n for n,p in params]:raise ValueError('Adapter order mismatch.')
                delta=torch.cat([p.detach().float().cpu().flatten() for n,p in params])-initial
            if plan is None:
                plan=[];sep=tok(str(model.config.data.separator),add_special_tokens=False).input_ids
                for r in pairs:
                    sid=r['sample_id'];prefix=[tok.bos_token_id]+tok(questions[sid]['question'].strip(),add_special_tokens=False).input_ids+sep
                    tokens=torch.tensor(r['tokens'])
                    if tokens[:len(prefix)].tolist()!=prefix:raise ValueError('Prefix mismatch.')
                    valid=completion_mask(tokens[None],torch.tensor([len(prefix)]),tok.eos_token_id,tok.pad_token_id,model.mask_index)[0]
                    for rep in range(4):
                        seed=stable_seed('partial75-objective',sid,int(r['reward']),rep)
                        mask=partial_completion_mask(valid[None],.75,seed)[0]
                        noisy=tokens.clone();noisy[len(prefix):]=model.mask_index;noisy[valid & ~mask]=tokens[valid & ~mask]
                        plan.append(dict(sample_id=sid,reward=r['reward'],replicate=rep,seed=seed,tokens=tokens,noisy=noisy,mask=mask))
                torch.save(plan,run/'fixed_plan.pt')
                (run/'plan_sha256.txt').write_text(hashlib.sha256((run/'fixed_plan.pt').read_bytes()).hexdigest())
            # A CPU-fixed mask plan is shared exactly by both models; do not compare GPU RNG streams.
            for reward in [1,0]:
                items=[x for x in plan if x['reward']==reward];denom=sum(int(x['mask'].sum()) for x in items)
                model.zero_grad(set_to_none=True)
                for start in range(0,len(items),8):
                    batch=items[start:start+8];xt=torch.stack([x['noisy'] for x in batch]).cuda();target=torch.stack([x['tokens'] for x in batch]).cuda();mask=torch.stack([x['mask'] for x in batch]).cuda()
                    _,alpha=model.noise(torch.ones(len(batch),1,device=model.device));sigma=model._sigma_from_alphat(alpha)
                    logps=model.forward(xt=xt,sigma=sigma);selected=logps.gather(-1,target[:,:,None]).squeeze(-1)
                    if logps.dtype!=torch.float32:raise ValueError('FP32 inactive.')
                    loss=-((reward-.5)*selected[mask]).sum()/denom;loss.backward()
                    for j,x in enumerate(batch):
                        value=float(selected[j][mask[j]].detach().mean())
                        scores.append(dict(model=label,sample_id=x['sample_id'],reward=reward,replicate=x['replicate'],mean_logp=value,ce=-value,top1=float(logps[j].argmax(-1)[mask[j]].eq(target[j][mask[j]]).float().mean()),tokens=int(mask[j].sum())))
                    print(json.dumps(dict(model=label,reward=reward,scored=min(start+8,len(items)),total=len(items))),flush=True)
                vectors[(label,reward)]=flatten_gradients([p for n,p in params]).cpu()
            del model;gc.collect();torch.cuda.empty_cache()
        (run/'scores.json').write_text(json.dumps(scores,indent=2))
        averaged={}
        for label in ['base','partial250']:
            for reward in [1,0]:
                out=[]
                for sid in questions:
                    rows=[r for r in scores if r['model']==label and r['reward']==reward and r['sample_id']==sid]
                    out.append(dict(sample_id=sid,**{k:sum(x[k] for x in rows)/len(rows) for k in ['mean_logp','ce','top1']}))
                averaged[(label,reward)]=out
        comparison={str(r):paired_summary(averaged[('base',r)],averaged[('partial250',r)]) for r in [1,0]}
        basegap=[dict(sample_id=s['sample_id'],mean_logp=s['mean_logp']-f['mean_logp'],ce=0,top1=0) for s,f in zip(averaged[('base',1)],averaged[('base',0)])]
        newgap=[dict(sample_id=s['sample_id'],mean_logp=s['mean_logp']-f['mean_logp'],ce=0,top1=0) for s,f in zip(averaged[('partial250',1)],averaged[('partial250',0)])]
        comparison['gap']=paired_summary(basegap,newgap)
        geometry={}
        for reward in [1,0,'balanced']:
            b=vectors[('base',reward)] if reward!='balanced' else (vectors[('base',1)]+vectors[('base',0)])/2
            c=vectors[('partial250',reward)] if reward!='balanced' else (vectors[('partial250',1)]+vectors[('partial250',0)])/2
            def cosine(x,y):
                d=float(x.norm()*y.norm());return float(torch.dot(x,y))/d if d else None
            geometry[str(reward)]=dict(base_gradient_norm=float(b.norm()),current_gradient_norm=float(c.norm()),gradient_cosine=cosine(b,c),update_cosine_negative_base_gradient=cosine(delta,-b),update_cosine_negative_current_gradient=cosine(delta,-c),base_first_order_loss_change=float(torch.dot(b,delta)))
        (run/'comparison.json').write_text(json.dumps(dict(scores=comparison,geometry=geometry,update_norm=float(delta.norm()),limitations=['Read-only; no optimizer steps.','Four fixed random masks/program; question bootstrap; one training seed.','Signed token-mean surrogate at old=current, A=reward-.5; no clipping/KL in gradient diagnostic.','Success/failure gradients normalized within each stratum; balanced gradient equally averages strata.','Net adapter displacement alignment is a local first-order diagnostic, not proof of optimization trajectory.','CPU masks follow the same 75% rule but are fresh fixed held-out masks, not replayed training masks.']),indent=2))
        torch.save(dict(gradients=vectors,adapter_displacement=delta,names=names),run/'gradient_vectors.pt')
        (run/'COMPLETE').touch();(run/'STAGE').write_text('complete\n')
    except BaseException:
        (run/'FAILED').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
