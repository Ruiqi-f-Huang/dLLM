#!/usr/bin/env python3
"""Fixed validation coverage comparison: old-seed8 plus fresh Base/8/32; read-only."""
import argparse,json,sys,traceback,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 p=argparse.ArgumentParser()
 for n in ['base','checkpoint','checkpoint8','data','previous-validation','output']:p.add_argument('--'+n,required=True)
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 (out/'run.json').write_text(json.dumps(vars(a),indent=2))
 try:
  import torch
  from grpo_fixed32 import read_jsonl,load_model
  from grpo_overfit import stable_seed,summarize_samples,paired_bootstrap
  from train_mdlm_grpo import execution_reward
  rows=read_jsonl(a.data)
  if len(rows)!=128 or len({r['sample_id'] for r in rows})!=128:raise ValueError('Expected frozen128.')
  previous=Path(a.previous_validation)
  if not (previous/'COMPLETE').exists():raise ValueError('Previous validation incomplete.')
  previous_args=json.loads((previous/'run.json').read_text())
  if Path(previous_args['data']).read_bytes()!=Path(a.data).read_bytes():raise ValueError('Validation identity mismatch.')
  old=json.loads((previous/'sampling.json').read_text())
  summaries={}
  plan=[('original-positive8','positive8',a.checkpoint8,20261024),
        ('fresh-base','base',None,20261025),('fresh-positive8','positive8',a.checkpoint8,20261025),
        ('fresh-positive32','positive32',a.checkpoint,20261025)]
  for stage,label,path,eval_seed in plan:
   (out/'STAGE').write_text(stage+'\n');print(stage,flush=True)
   summaries[stage]={}
   model,tok=load_model(a.base,path);model.config.backbone_compute_dtype='float32';model.requires_grad_(False)
   sep=tok(str(model.config.data.separator),add_special_tokens=False).input_ids
   for temp in [.3,.1]:
    records=[]
    with (out/f'{stage}-t{temp:g}.jsonl').open('x') as log,torch.inference_mode():
     for r in rows:
      prefix=[tok.bos_token_id]+tok(r['question'].strip(),add_special_tokens=False).input_ids+sep
      if len(prefix)>=model.num_tokens:raise ValueError('Prefix too long.')
      for start in [0,8]:
       seed=stable_seed('positive32-validation',eval_seed,r['sample_id'],temp,start)
       torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
       ids=torch.tensor([prefix],device=model.device).expand(8,-1);lengths=torch.full((8,),len(prefix),device=model.device,dtype=torch.long)
       model.sampler.temperature=temp
       tokens,_=model.generate_samples(8,num_steps=32,prefix_tokens=ids,prefix_lengths=lengths)
       for j,seq in enumerate(tokens.cpu().tolist()):
        suffix=seq[len(prefix):]
        if tok.eos_token_id in suffix:suffix=suffix[:suffix.index(tok.eos_token_id)]
        text=tok.decode(suffix,skip_special_tokens=True);details=execution_reward(text,r['answer'],2.,details=True)
        rec=dict(sample_id=r['sample_id'],repeat=start+j,seed=seed,temperature=temp,reward=details['reward'],execution=details,text=text)
        log.write(json.dumps(rec,ensure_ascii=False)+'\n');records.append(rec)
       log.flush()
      print(json.dumps(dict(stage=stage,model=label,temperature=temp,draws=len(records),total=2048)),flush=True)
    summary=summarize_samples(records,16)
    summary['raw_correct']=sum(r['execution'].get('raw',{}).get('correct',False) for r in records)
    summary['infrastructure_failures']=sum(r['execution'].get('infrastructure_failure',False) for r in records)
    summaries[stage][str(temp)]=summary
    (out/'sampling.json').write_text(json.dumps(summaries,indent=2))
   del model;gc.collect();torch.cuda.empty_cache()
  batches={'original':{'base':old['base'],'positive32':old['positive32'],'positive8':summaries['original-positive8']},
           'fresh':{name:summaries['fresh-'+name] for name in ['base','positive8','positive32']}}
  combined={}
  for label in ['base','positive8','positive32']:
   combined[label]={}
   for t in ['0.3','0.1']:
    first={r['sample_id']:r for r in batches['original'][label][t]['per_question']}
    second={r['sample_id']:r for r in batches['fresh'][label][t]['per_question']}
    if first.keys()!=second.keys():raise ValueError('Question mismatch.')
    rows=[]
    for sid in sorted(first):
     n=first[sid]['draws']+second[sid]['draws'];hits=first[sid]['correct']+second[sid]['correct']
     rows.append(dict(sample_id=sid,correct=hits,draws=n,success_probability=hits/n))
    combined[label][t]=dict(per_question=rows,draws=sum(r['draws'] for r in rows),success_at_1=sum(r['success_probability'] for r in rows)/len(rows))
  batches['combined']=combined
  comparisons={}
  for batch,models in batches.items():
   comparisons[batch]={}
   for left,right in [('base','positive8'),('base','positive32'),('positive8','positive32')]:
    comparisons[batch][left+'-to-'+right]={t:paired_bootstrap(models[left][t],models[right][t]) for t in ['0.3','0.1']}
  (out/'comparison.json').write_text(json.dumps(comparisons,indent=2))
  (out/'all_batches.json').write_text(json.dumps(batches,indent=2))
  (out/'limitations.json').write_text(json.dumps(['No new training; one checkpoint per fit.','Same frozen128 validation, not an additional held-out dataset.','Fresh seed is independent of original seed; identical seed protocol across models.','Combined bootstrap clusters by question; do not treat repeats as independent questions.','Coverage and total successful-program training compute differ between fixed8 and fixed32.'],indent=2))
  (out/'STAGE').write_text('complete\n')
  (out/'COMPLETE').touch()
 except BaseException:
  (out/'FAILED').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
