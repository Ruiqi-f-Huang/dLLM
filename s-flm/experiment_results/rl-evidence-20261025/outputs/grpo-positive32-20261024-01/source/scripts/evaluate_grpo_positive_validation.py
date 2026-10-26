#!/usr/bin/env python3
"""Frozen validation generation only; no pair selection or optimizer."""
import argparse,json,sys,traceback,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 p=argparse.ArgumentParser()
 for n in ['base','checkpoint','data','output']:p.add_argument('--'+n,required=True)
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 (out/'run.json').write_text(json.dumps(vars(a),indent=2))
 try:
  import torch
  from grpo_fixed32 import read_jsonl,load_model
  from grpo_overfit import stable_seed,summarize_samples,paired_bootstrap
  from train_mdlm_grpo import execution_reward
  rows=read_jsonl(a.data)
  if len(rows)!=128 or len({r['sample_id'] for r in rows})!=128:raise ValueError('Expected frozen128.')
  summaries={}
  for label,path in [('base',None),('positive32',a.checkpoint)]:
   model,tok=load_model(a.base,path);model.config.backbone_compute_dtype='float32';model.requires_grad_(False)
   sep=tok(str(model.config.data.separator),add_special_tokens=False).input_ids;summaries[label]={}
   for temp in [.3,.1]:
    records=[]
    with (out/f'{label}-t{temp:g}.jsonl').open('x') as log,torch.inference_mode():
     for r in rows:
      prefix=[tok.bos_token_id]+tok(r['question'].strip(),add_special_tokens=False).input_ids+sep
      if len(prefix)>=model.num_tokens:raise ValueError('Prefix too long.')
      for start in [0,8]:
       seed=stable_seed('positive32-validation',20261024,r['sample_id'],temp,start)
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
      print(json.dumps(dict(model=label,temperature=temp,draws=len(records),total=2048)),flush=True)
    summary=summarize_samples(records,16)
    summary['raw_correct']=sum(r['execution'].get('raw',{}).get('correct',False) for r in records)
    summary['infrastructure_failures']=sum(r['execution'].get('infrastructure_failure',False) for r in records)
    summaries[label][str(temp)]=summary
    (out/'sampling.json').write_text(json.dumps(summaries,indent=2))
   del model;gc.collect();torch.cuda.empty_cache()
  (out/'comparison.json').write_text(json.dumps({t:paired_bootstrap(summaries['base'][t],summaries['positive32'][t]) for t in ['0.3','0.1']},indent=2))
  (out/'COMPLETE').touch()
 except BaseException:
  (out/'FAILED').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
