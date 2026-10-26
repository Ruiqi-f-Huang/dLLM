#!/usr/bin/env python3
"""Frozen three-way full GSM8K Test comparison; T=.1, NFE32, seeds1/2, FP32."""
import argparse,json,sys,time,hashlib,gc,traceback,math,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SFT_SHA='5b2f034c078a7557814cb9d243387ca525f9cb418568556740a569a23427029a'
def read_existing(path,rows,seed):
 records=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
 if len(records)>len(rows):raise ValueError('Too many saved records.')
 from grpo_overfit import stable_seed
 for i,r in enumerate(records):
  if any(r.get(k)!=rows[i][k] for k in ['sample_id','question','answer']) or r.get('seed')!=seed or r.get('batch_seed')!=stable_seed('final-gsm8k-test-v1',seed,(i//8)*8):raise ValueError('Saved record does not match frozen question/seed order.')
  if r['execution'].get('infrastructure_failure') or r['reward'] not in [0,1]:raise ValueError('Invalid saved reward.')
 return records

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['base','sft','positive','test','output']:p.add_argument('--'+n,required=True)
 p.add_argument('--preflight-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
 paths={n:Path(getattr(a,n)).resolve() for n in ['base','sft','positive','test']}
 missing=[str(v) for v in paths.values() if not v.is_file()]
 if missing:raise FileNotFoundError('Missing required files: '+str(missing))
 raw=json.loads(paths['test'].read_text())
 if len(raw)!=1319 or len({r['prompt'] for r in raw})!=1319:raise ValueError('Expected1319 unique test questions.')
 if any(not str(r['response_ground_truth']).strip() or not str(r['prompt']).strip() for r in raw):raise ValueError('Missing question/answer.')
 hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}
 if hashes['sft']!=SFT_SHA:raise ValueError('SFT711 checksum differs from archived manifest.')
 import torch
 sft=torch.load(paths['sft'],map_location='cpu',weights_only=False)
 if sft.get('global_step')!=711 or sft.get('ema',{}).get('num_updates')!=711 or len(sft['ema']['shadow_params'])!=96:raise ValueError('Wrong SFT step/EMA.')
 del sft;gc.collect()
 if a.preflight_only:print(json.dumps(dict(status='preflight-passed',hashes=hashes)));return
 out=Path(a.output).resolve()
 if a.resume:
  frozen=json.loads((out/'protocol.json').read_text())
  if frozen['hashes']!=hashes or frozen['seeds']!=[1,2] or frozen['temperature']!=.1 or frozen['nfe']!=32 or frozen['batch_size']!=8 or frozen['precision']!='float32':raise ValueError('Cannot resume a different protocol.')
  if (out/'COMPLETE').exists():raise ValueError('Run already complete.')
 else:out.mkdir(parents=True,exist_ok=False)
 (out/('runner-resume.py' if a.resume else 'runner.py')).write_text(Path(__file__).read_text())
 protocol=dict(models=['base_ema','sft711_ema','positive32_raw'],questions=1319,temperature=.1,nfe=32,precision='float32',seeds=[1,2],batch_size=8,primary='mean single-draw execution accuracy across two fixed seeds',test_history='Test previously used in exploratory evaluations; not untouched.',weights='Base EMA import; SFT711 EMA copy and verify; positive32 raw',frozen_before_generation=True)
 if not a.resume:(out/'protocol.json').write_text(json.dumps(dict(**protocol,paths={k:str(v) for k,v in paths.items()},hashes=hashes),indent=2))
 try:
  from grpo_fixed32 import load_model
  from grpo_overfit import stable_seed,paired_bootstrap
  from train_mdlm_grpo import execution_reward
  rows=[dict(sample_id=f'gsm8k_test_{i:04d}',question=r['prompt'],answer=r['response_ground_truth']) for i,r in enumerate(raw)]
  if not a.resume:(out/'questions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
  summaries=json.loads((out/'sampling.json').read_text()) if a.resume else {}
  reference_audit=json.loads((out/'base_ema-sampler-audit.json').read_text()) if a.resume else None
  if a.resume and (out/'FAILED').exists():(out/'FAILED').rename(out/f'FAILED-before-resume-{time.time_ns()}')
  for label,path in [('base_ema',None),('sft711_ema',str(paths['sft'])),('positive32_raw',str(paths['positive']))]:
   existing={seed:read_existing(out/f'{label}-seed{seed}.jsonl',rows,seed) for seed in [1,2]} if a.resume else {1:[],2:[]}
   for seed,rs in existing.items():
    if len(rs)==1319:
     ss=summaries.get(label,{}).get(str(seed))
     if ss is None or ss['correct']!=sum(r['reward'] for r in rs):raise ValueError('Completed records/summary mismatch.')
   if all(len(rs)==1319 for rs in existing.values()):
    print('RESUME_SKIP_COMPLETED',label,flush=True);continue
   (out/'STAGE').write_text(label+'\n');print(label,flush=True)
   model,tok=load_model(str(paths['base']),path);model.config.backbone_compute_dtype='float32'
   weight_info=dict(mode='base_ema_import' if label=='base_ema' else 'raw')
   if label=='sft711_ema':
    if model.ema is None:raise ValueError('SFT EMA unavailable.')
    params=[p for p in model._get_parameters() if p.requires_grad]
    shadows=model.ema.shadow_params
    if len(params)!=96 or len(shadows)!=96 or any(p.shape!=v.shape for p,v in zip(params,shadows)):raise ValueError('EMA order/shape mismatch.')
    model.ema.copy_to(model._get_parameters())
    if any(not torch.equal(p.detach().cpu(),v.cpu()) for p,v in zip(params,shadows)):raise ValueError('EMA copy verification failed.')
    weight_info=dict(mode='ema',tensors=96,updates=model.ema.num_updates,verified=True)
   (out/(label+'-weights.json')).write_text(json.dumps(weight_info,indent=2))
   model.requires_grad_(False);model.eval();model.sampler.temperature=.1
   sep=tok(str(model.config.data.separator),add_special_tokens=False).input_ids
   prefixes=[[tok.bos_token_id]+tok(r['question'].strip(),add_special_tokens=False).input_ids+sep for r in rows]
   if any(len(x)>=model.num_tokens for x in prefixes):raise ValueError('Question too long for fixed context; no silent truncation.')
   from samplers import AncestralSampler
   if not isinstance(model.sampler,AncestralSampler):raise ValueError('Expected ancestral sampler.')
   audit=dict(sampler=type(model.sampler).__name__,steps_policy=model.sampler.steps_policy,noise_removal=model.sampler.noise_removal,use_float64=model.sampler.use_float64,temperature=model.sampler.temperature,eps=1e-5,nfe=32,num_tokens=model.num_tokens,diffusion_type=model.diffusion_type,tokenizer_vocab_sha256=hashlib.sha256(json.dumps(tok.get_vocab(),sort_keys=True).encode()).hexdigest(),special_ids=[tok.bos_token_id,tok.eos_token_id,tok.pad_token_id,model.mask_index],separator_ids=sep,prefix_sha256=hashlib.sha256(json.dumps(prefixes).encode()).hexdigest())
   if (audit['steps_policy'],audit['noise_removal'],audit['use_float64'],audit['num_tokens'])!=('full','ancestral',True,512):raise ValueError('Sampling configuration differs from frozen protocol: '+str(audit))
   if reference_audit is None:reference_audit=audit
   elif audit!=reference_audit:raise ValueError('Sampling/tokenization differs between models.')
   (out/(label+'-sampler-audit.json')).write_text(json.dumps(audit,indent=2))
   def verify_backbone(module,inputs,result):
    if result.dtype!=torch.float32:raise ValueError('Actual backbone output must be FP32, got '+str(result.dtype))
    (out/(label+'-precision.json')).write_text(json.dumps(dict(backbone_output_dtype=str(result.dtype))))
    precision_hook.remove()
   precision_hook=model.backbone.register_forward_hook(verify_backbone)
   torch.cuda.reset_peak_memory_stats();t0=time.monotonic();summaries.setdefault(label,{})
   for seed in [1,2]:
    if len(existing[seed])==1319:continue
    seed0=time.monotonic();records=list(existing[seed]);saved_count=len(records)
    with (out/f'{label}-seed{seed}.jsonl').open('a' if a.resume else 'x') as log,torch.inference_mode():
     for start in range((saved_count//8)*8,len(rows),8):
      batch=rows[start:start+8];pr=prefixes[start:start+8];lengths=torch.tensor([len(x) for x in pr],device=model.device)
      ids=torch.full((len(batch),max(map(len,pr))),tok.pad_token_id,device=model.device,dtype=torch.long)
      for j,x in enumerate(pr):ids[j,:len(x)]=torch.tensor(x,device=model.device)
      rngseed=stable_seed('final-gsm8k-test-v1',seed,start);torch.manual_seed(rngseed);torch.cuda.manual_seed_all(rngseed)
      tokens,metadata=model.generate_samples(len(batch),num_steps=32,eps=1e-5,prefix_tokens=ids,prefix_lengths=lengths)
      if metadata.get('nfe')!=32:raise ValueError('Unexpected actual NFE: '+str(metadata))
      for j,seq in enumerate(tokens.cpu().tolist()):
       if seq[:len(pr[j])]!=pr[j]:raise ValueError('Sampler modified the question prefix.')
       suffix=seq[len(pr[j]):];eos=tok.eos_token_id in suffix
       if eos:suffix=suffix[:suffix.index(tok.eos_token_id)]
       text=tok.decode(suffix,skip_special_tokens=True)
       if start+j<saved_count:
        if records[start+j]['text']!=text:raise ValueError('Regenerated partial batch differs; refusing to mix sampling streams.')
        continue
       pending=dict(model=label,seed=seed,index=start+j,batch_seed=rngseed,text=text,answer=batch[j]['answer'])
       (out/'pending-evaluation.json').write_text(json.dumps(pending,ensure_ascii=False))
       for retry in range(3):
        details=execution_reward(text,batch[j]['answer'],2.,details=True)
        if not details.get('infrastructure_failure'):break
       if details.get('infrastructure_failure'):
        (out/'evaluation-failure.json').write_text(json.dumps(dict(**pending,execution=details),ensure_ascii=False))
        raise RuntimeError('Evaluator infrastructure failed after3 retries; preserve output and report.')
       record=dict(sample_id=batch[j]['sample_id'],seed=seed,batch_seed=rngseed,question=batch[j]['question'],answer=batch[j]['answer'],text=text,reward=details['reward'],execution=details,eos_found=eos,completion_length=len(suffix),infrastructure_retries=retry)
       log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush();records.append(record)
      log.flush();print(json.dumps(dict(model=label,seed=seed,finished=len(records),total=1319)),flush=True)
    summary=dict(questions=1319,correct=sum(x['reward'] for x in records),accuracy=sum(x['reward'] for x in records)/1319,elapsed_seconds=time.monotonic()-seed0,raw_correct=sum(x['execution']['raw']['correct'] for x in records),infrastructure_retries=sum(x['infrastructure_retries'] for x in records),mean_completion_length=sum(x['completion_length'] for x in records)/1319,per_question=[dict(sample_id=x['sample_id'],correct=x['reward'],draws=1,success_probability=x['reward']) for x in records])
    summary['errors']={k:sum(x['execution']['raw'].get(k,False) for x in records) for k in ['parses','executes','numeric','correct']}
    if saved_count:summary['elapsed_scope']='resume session only; original partial-session time unavailable'
    summaries[label][str(seed)]=summary;(out/'sampling.json').write_text(json.dumps(summaries,indent=2))
   (out/(label+('-resume-cost.json' if a.resume else '-cost.json'))).write_text(json.dumps(dict(evaluation_seconds=time.monotonic()-t0,scope='resume session only' if a.resume else 'complete model evaluation',peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu=torch.cuda.get_device_name()),indent=2))
   del model;gc.collect();torch.cuda.empty_cache()
  combined={}
  for label,ss in summaries.items():
   byseed=[{r['sample_id']:r for r in ss[str(seed)]['per_question']} for seed in [1,2]]
   combined[label]=dict(per_question=[dict(sample_id=sid,correct=byseed[0][sid]['correct']+byseed[1][sid]['correct'],draws=2,success_probability=(byseed[0][sid]['correct']+byseed[1][sid]['correct'])/2) for sid in sorted(byseed[0])])
  comparison={}
  for left,right in [('base_ema','sft711_ema'),('base_ema','positive32_raw'),('sft711_ema','positive32_raw')]:
   comparison[left+'-to-'+right]=dict(seeds={str(seed):paired_bootstrap(summaries[left][str(seed)],summaries[right][str(seed)]) for seed in [1,2]},combined=paired_bootstrap(combined[left],combined[right]))
  (out/'comparison.json').write_text(json.dumps(comparison,indent=2));(out/'combined.json').write_text(json.dumps(combined,indent=2))
  (out/'COMPLETE').touch();(out/'STAGE').write_text('complete\n')
 except BaseException:
  (out/'FAILED').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
