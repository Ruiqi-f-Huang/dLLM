#!/usr/bin/env python3
"""Collect Base successes for96 additional train questions, fit128, evaluate fixed632."""
import argparse,json,sys,subprocess,traceback,gc,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 p=argparse.ArgumentParser()
 for n in ['base','source32','train-data','validation','output']:p.add_argument('--'+n,required=True)
 a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 (out/'run.json').write_text(json.dumps(dict(**vars(a),candidate_budget_per_question=8,max_candidate_questions=512,temperature=.3,nfe=32),indent=2))
 def stage(s):(out/'STAGE').write_text(s+'\n')
 def execute(command,label):
  stage(label);print(label,flush=True)
  with (out/(label+'.log')).open('x') as log:subprocess.run([sys.executable,'-u']+command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 try:
  import torch
  from grpo_fixed32 import read_jsonl,load_model
  from grpo_overfit import stable_seed
  from train_mdlm_grpo import execution_reward
  source=Path(a.source32)
  if not (source/'COMPLETE').exists() or not (source/'train/mask_plan.pt').exists():raise ValueError('Source32 incomplete.')
  old_questions=read_jsonl(source/'questions.jsonl');pairs=read_jsonl(source/'pairs.jsonl');train=read_jsonl(a.train_data);validation=read_jsonl(a.validation)
  norm=lambda s:' '.join(s.casefold().split())
  trainids={r['sample_id'] for r in train};traintexts={norm(r['question']) for r in train}
  if trainids & {r['sample_id'] for r in validation} or traintexts & {norm(r['question']) for r in validation}:raise ValueError('Train/validation overlap.')
  if len(validation)!=632:raise ValueError('Expected full632 development validation.')
  (out/'validation632.jsonl').write_text(Path(a.validation).read_text())
  oldids={r['sample_id'] for r in old_questions};oldtexts={norm(r['question']) for r in old_questions}
  if len(oldids)!=32 or not oldids<=trainids:raise ValueError('Original32 not training subset.')
  candidates=[r for r in train if r['sample_id'] not in oldids and norm(r['question']) not in oldtexts]
  candidates.sort(key=lambda r:(stable_seed('positive128-training-selection',42,r['sample_id']),r['sample_id']))
  seenids=set(oldids);seentexts=set(oldtexts);order=[]
  for r in candidates:
   if r['sample_id'] in seenids or norm(r['question']) in seentexts:continue
   order.append(r);seenids.add(r['sample_id']);seentexts.add(norm(r['question']))
  order=order[:512]
  (out/'candidate_order.json').write_text(json.dumps([r['sample_id'] for r in order],indent=2))
  (out/'data_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [a.train_data,a.validation,str(source/'pairs.jsonl'),str(source/'train/mask_plan.pt')]},indent=2))
  for name in ['scripts/run_grpo_positive128.py','scripts/train_grpo_positive128.py','scripts/evaluate_grpo_positive128.py','diffu_grpo.py']:
   dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
  stage('collect-successes');print('collect-successes',flush=True)
  model,tok=load_model(a.base,None);model.config.backbone_compute_dtype='float32';model.requires_grad_(False);model.sampler.temperature=.3
  sep=tok(str(model.config.data.separator),add_special_tokens=False).input_ids
  def prefix(r):return [tok.bos_token_id]+tok(r['question'].strip(),add_special_tokens=False).input_ids+sep
  def decoded(ids,length):
   suffix=ids[length:]
   if tok.eos_token_id in suffix:suffix=suffix[:suffix.index(tok.eos_token_id)]
   return tok.decode(suffix,skip_special_tokens=True)
  def checked_reward(text,answer):
   for attempt in range(3):
    result=execution_reward(text,answer,2.,details=True)
    if not result.get('infrastructure_failure'):
     result['infrastructure_retries']=attempt;return result
   raise RuntimeError('Execution infrastructure failed after3 attempts.')
  programs=[];selected=list(old_questions)
  for r in old_questions:
   matches=[x for x in pairs if x['sample_id']==r['sample_id'] and x['reward']==1]
   if len(matches)!=1:raise ValueError('Missing original success.')
   ids=matches[0]['tokens'];pr=prefix(r)
   if ids[:len(pr)]!=pr or len(ids)!=model.num_tokens:raise ValueError('Original token mismatch.')
   details=checked_reward(decoded(ids,len(pr)),r['answer'])
   if details.get('infrastructure_failure') or details['reward']!=1:raise ValueError('Original success revalidation failed.')
   programs.append(dict(sample_id=r['sample_id'],reward=1.,tokens=ids,prefix_length=len(pr),origin='retained32',execution=details))
  print('original32-successes-revalidated',flush=True)
  attempted=0;success_questions=0;skipped=0
  with (out/'candidate_rollouts.jsonl').open('x') as log:
   for r in order:
    if len(programs)==128:break
    pr=prefix(r)
    if len(pr)>=model.num_tokens:skipped+=1;continue
    seed=stable_seed('positive128-base-candidates',20261025,r['sample_id']);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    ids=torch.tensor([pr],device=model.device).expand(8,-1);lengths=torch.full((8,),len(pr),device=model.device,dtype=torch.long)
    with torch.inference_mode():tokens,_=model.generate_samples(8,num_steps=32,prefix_tokens=ids,prefix_lengths=lengths)
    chosen=None;attempted+=1
    for j,seq in enumerate(tokens.cpu().tolist()):
     text=decoded(seq,len(pr));details=checked_reward(text,r['answer'])
     if details.get('infrastructure_failure'):raise RuntimeError('Candidate execution infrastructure failure; do not treat as a model failure.')
     record=dict(sample_id=r['sample_id'],candidate=j,seed=seed,reward=details['reward'],execution=details,text=text,tokens=seq,prefix_length=len(pr))
     log.write(json.dumps(record,ensure_ascii=False)+'\n')
     if chosen is None and details['reward']==1:chosen=dict(record,origin='new96')
    log.flush()
    if chosen is not None:programs.append(chosen);selected.append(r);success_questions+=1
    progress=dict(attempted_questions=attempted,successful_questions=success_questions,selected_programs=len(programs),target=128,skipped_prefix=skipped)
    (out/'collection_progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
  if len(programs)!=128:raise ValueError('Candidate budget exhausted before128; no automatic budget increase.')
  (out/'programs128.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in programs))
  (out/'questions128.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
  (out/'collection_summary.json').write_text(json.dumps(dict(retained32=32,new96=96,attempted_questions=attempted,success_fraction=success_questions/attempted,candidates_per_question=8,selection='fixed training hash order; first correct candidate; no validation selection',precision='float32'),indent=2))
  del model;gc.collect();torch.cuda.empty_cache()
  execute(['scripts/train_grpo_positive128.py','--base',a.base,'--programs',str(out/'programs128.jsonl'),'--original-plan',str(source/'train/mask_plan.pt'),'--output',str(out/'train')],'train')
  execute(['scripts/evaluate_grpo_positive128.py','--base',a.base,'--checkpoint',str(out/'train/positive128-step200.ckpt'),'--checkpoint32',str(source/'positive32-step200.ckpt'),'--data',str(out/'validation632.jsonl'),'--output',str(out/'validation')],'validation')
  (out/'limitations.json').write_text(json.dumps(['Positive rollout distillation, not online GRPO.','Original32 successes retained from prior Base generation; new96 freshly generated FP32.','Full632 validation is a development set already used in this project, not a fresh test.','No Base evaluation; compare32 versus128 on same A800 and same evaluation seed.','Coverage and total successful-program compute increase together; per-program updates remain200.'],indent=2))
  stage('complete');(out/'COMPLETE').touch()
 except BaseException:
  (out/'FAILED').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
