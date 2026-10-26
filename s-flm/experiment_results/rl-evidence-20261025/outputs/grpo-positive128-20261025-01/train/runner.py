#!/usr/bin/env python3
"""Fixed128 success-only partial75 fit, preserving original32 masks and loss scale."""
import argparse,json,sys,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 p=argparse.ArgumentParser()
 for n in ['base','programs','original-plan','output']:p.add_argument('--'+n,required=True)
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 (out/'run.json').write_text(json.dumps(dict(**vars(a),lr=3e-5,steps=200,loss_scale=.25),indent=2));(out/'runner.py').write_text(Path(__file__).read_text())
 try:
  import torch
  import lightning as L
  from grpo_fixed32 import load_model,read_jsonl
  from diffu_grpo import completion_mask,partial_completion_mask
  from grpo_overfit import stable_seed
  model,tok=load_model(a.base,None);model.config.backbone_compute_dtype='float32'
  records=read_jsonl(a.programs)
  if len(records)!=128 or len({r['sample_id'] for r in records})!=128 or any(r['reward']!=1 for r in records):raise ValueError('Expected128 successes.')
  original=torch.load(a.original_plan,map_location='cpu',weights_only=False)
  old={r['sample_id']:j for j,r in enumerate(original['metadata']) if r['reward']==1}
  if len(old)!=32 or not set(old)<={r['sample_id'] for r in records}:raise ValueError('Original32 missing.')
  tokens=torch.tensor([r['tokens'] for r in records],device=model.device)
  lengths=torch.tensor([r['prefix_length'] for r in records],device=model.device)
  valid=completion_mask(tokens,lengths,tok.eos_token_id,tok.pad_token_id,model.mask_index)
  masks=[[],[],[],[],[]]
  for i,r in enumerate(records):
   sid=r['sample_id']
   if sid in old:
    j=old[sid]
    if not torch.equal(tokens[i].cpu(),original['tokens'][j]):raise ValueError('Original program changed.')
    chosen=[original['train_mask'][j]]+[x[j] for x in original['heldout_masks']]
   else:chosen=[partial_completion_mask(valid[i:i+1].cpu(),.75,stable_seed('positive128-mask',sid,k))[0] for k in range(5)]
   for k,m in enumerate(chosen):
    if not m.any() or (m & ~valid[i].cpu()).any():raise ValueError('Invalid mask.')
    masks[k].append(m)
  masks=[torch.stack(x).to(model.device) for x in masks]
  suffix=torch.arange(tokens.shape[1],device=model.device)[None,:]>=lengths[:,None]
  inputs=[]
  for m in masks:
   xt=tokens.masked_fill(suffix,model.mask_index);xt=torch.where(valid & ~m,tokens,xt);inputs.append(xt)
  torch.save(dict(tokens=tokens.cpu(),lengths=lengths.cpu(),masks=[m.cpu() for m in masks],metadata=[dict(sample_id=r['sample_id'],original32=r['sample_id'] in old) for r in records]),out/'mask_plan.pt')
  params=[p for p in model.parameters() if p.requires_grad]
  initial=[p.detach().clone() for p in params]
  frozen={n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in model.named_parameters() if not p.requires_grad}
  optimizer=torch.optim.AdamW(params,lr=3e-5,weight_decay=0.)
  def logps(sl,k):
   xt=inputs[k][sl];_,alpha=model.noise(torch.ones(len(xt),1,device=model.device))
   prediction=model.forward(xt=xt,sigma=model._sigma_from_alphat(alpha))
   if prediction.dtype!=torch.float32:raise ValueError('FP32 inactive.')
   return prediction.gather(-1,tokens[sl,:,None]).squeeze(-1)
  @torch.no_grad()
  def measure(step):
   result=dict(step=step,parameter_delta_norm=sum((p-i).float().square().sum().item() for p,i in zip(params,initial))**.5,mask_scores=[])
   for k in range(5):
    values=[]
    for start in range(0,128,2):
     sl=slice(start,start+2);lp=logps(sl,k);m=masks[k][sl]
     values.extend((torch.where(m,lp,0.).sum(1)/m.sum(1)).tolist())
    result['mask_scores'].append(dict(mask=k,mean_logp=sum(values)/128,original32_mean_logp=sum(v for r,v in zip(records,values) if r['sample_id'] in old)/32,added96_mean_logp=sum(v for r,v in zip(records,values) if r['sample_id'] not in old)/96,per_program=[dict(sample_id=r['sample_id'],mean_logp=v) for r,v in zip(records,values)]))
   if not all(__import__('math').isfinite(x['mean_logp']) for x in result['mask_scores']):raise FloatingPointError('Invalid score.')
   print(json.dumps(dict(step=step,train_logp=result['mask_scores'][0]['mean_logp'],heldout_logp=sum(x['mean_logp'] for x in result['mask_scores'][1:])/4)),flush=True)
   return result
  measurements=[]
  with (out/'scores.jsonl').open('x') as scores,(out/'updates.jsonl').open('x') as updates:
   def score(step):
    r=measure(step);measurements.append(r);scores.write(json.dumps(r)+'\n');scores.flush()
   score(0)
   for step in range(1,201):
    optimizer.zero_grad(set_to_none=True);total=0.
    for start in range(0,128,2):
     sl=slice(start,start+2);lp=logps(sl,0);m=masks[0][sl]
     loss=-.25*(torch.where(m,lp,0.).sum(1)/m.sum(1)).sum()/128
     if not torch.isfinite(loss):raise FloatingPointError('Invalid loss.')
     loss.backward();total+=float(loss.detach())
    norm=torch.nn.utils.clip_grad_norm_(params,.2,error_if_nonfinite=True);optimizer.step()
    updates.write(json.dumps(dict(step=step,loss_before_update=total,gradient_norm=float(norm)))+'\n');updates.flush()
    print(json.dumps(dict(update=step,loss=total,grad_norm=float(norm))),flush=True)
    if step in [1,2,10,50,100,200]:score(step)
    if step in [50,100,200]:torch.save(dict(adapters={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},step=step,args=vars(a)),out/f'adapters-step{step:06d}.pt')
   changed=[n for n,p in model.named_parameters() if not p.requires_grad and hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()!=frozen[n]]
   if changed:raise ValueError('Frozen parameters changed.')
   torch.save(dict(state_dict=model.state_dict(),hyper_parameters=dict(model.hparams),global_step=200,grpo_args=vars(a),**{'pytorch-lightning_version':L.__version__}),out/'positive128-step200.ckpt')
   (out/'summary.json').write_text(json.dumps(dict(initial=measurements[0],final=measurements[-1],frozen_changed=changed,objective='-.25 times mean successful-program masked-token logp; no KL/clipping'),indent=2))
  (out/'COMPLETE').touch()
 except BaseException:
  (out/'FAILED').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
