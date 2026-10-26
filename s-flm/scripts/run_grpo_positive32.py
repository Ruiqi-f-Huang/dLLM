#!/usr/bin/env python3
"""One expanded positive32 fit with frozen independent validation evaluation."""
import argparse,json,sys,subprocess,traceback,shutil,gc,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 p=argparse.ArgumentParser()
 for n in ['base','online-run','positive8-evaluation','validation','train-data','output']:p.add_argument('--'+n,required=True)
 a=p.parse_args();run=Path(a.output).resolve();run.mkdir(parents=True,exist_ok=False)
 (run/'run.json').write_text(json.dumps(vars(a),indent=2))
 def execute(command,label):
  (run/'STAGE').write_text(label+'\n');print(label,flush=True)
  with (run/(label+'.log')).open('x') as log:subprocess.run([sys.executable,'-u']+command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 try:
  from grpo_fixed32 import read_jsonl,load_model
  from grpo_overfit import stable_seed,paired_bootstrap
  online=Path(a.online_run);old=Path(a.positive8_evaluation)
  for d in [online,old]:
   if not (d/'COMPLETE').exists():raise ValueError('Incomplete source.')
  for name in ['questions.jsonl','pairs.jsonl']:shutil.copy2(online/name,run/name)
  train=read_jsonl(a.train_data);questions=read_jsonl(run/'questions.jsonl');validation=read_jsonl(a.validation)
  normalize=lambda x:' '.join(x.strip().casefold().split())
  trainids={r['sample_id'] for r in train+questions};traintexts={normalize(r['question']) for r in train+questions}
  seenids=set();seentexts=set();eligible=[]
  for r in sorted(validation,key=lambda r:(stable_seed('positive32-validation-selection',42,r['sample_id']),r['sample_id'])):
   text=normalize(r['question'])
   if r['sample_id'] in trainids or text in traintexts:raise ValueError('Train/validation overlap.')
   if r['sample_id'] in seenids or text in seentexts:continue
   seenids.add(r['sample_id']);seentexts.add(text);eligible.append(r)
  if len(eligible)<128:raise ValueError('Insufficient independent validation.')
  selected=eligible[:128]
  (run/'validation128.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
  (run/'validation_selection.json').write_text(json.dumps(dict(count=128,selection='fixed hash order, before optimization',source_sha256=hashlib.sha256(Path(a.validation).read_bytes()).hexdigest(),train_sha256=hashlib.sha256(Path(a.train_data).read_bytes()).hexdigest(),normalized_overlap=0),indent=2))
  for name in ['scripts/run_grpo_positive32.py','scripts/train_grpo_partial75_positive32.py','scripts/evaluate_grpo_positive_validation.py','diffu_grpo.py','grpo_controlled.py']:
   dest=run/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
  execute(['scripts/train_grpo_partial75_positive32.py','--base',a.base,'--questions',str(run/'questions.jsonl'),'--pairs',str(run/'pairs.jsonl'),'--output',str(run/'train'),'--precision','float32','--lr','3e-5','--steps','200','--count','32','--microbatch','2'],'train')
  import torch
  import lightning as L
  artifact=torch.load(run/'train/final_adapters.pt',map_location='cpu',weights_only=False)
  model,tok=load_model(a.base,None);state=model.state_dict();expected={n for n,p in model.named_parameters() if p.requires_grad}
  if expected!=set(artifact['adapters']):raise ValueError('Adapter keys differ.')
  for name,value in artifact['adapters'].items():
   if state[name].shape!=value.shape or not torch.isfinite(value).all():raise ValueError('Invalid adapter.')
   state[name]=value.to(state[name].device)
  model.load_state_dict(state,strict=True);model.config.backbone_compute_dtype='float32';checkpoint=run/'positive32-step200.ckpt'
  torch.save(dict(state_dict=model.state_dict(),hyper_parameters=dict(model.hparams),global_step=200,grpo_args=artifact['args'],**{'pytorch-lightning_version':L.__version__}),checkpoint)
  del model,state;gc.collect();torch.cuda.empty_cache()
  verify,_=load_model(a.base,str(checkpoint))
  if any(not torch.equal(p.detach().cpu(),artifact['adapters'][n]) for n,p in verify.named_parameters() if n in expected):raise ValueError('Adapter reload mismatch.')
  del verify,artifact;gc.collect();torch.cuda.empty_cache()
  execute(['scripts/grpo_fixed32.py','measure','--base',a.base,'--checkpoint',str(checkpoint),'--data',str(run/'questions.jsonl'),'--pairs',str(run/'pairs.jsonl'),'--precision','float32','--execution-details','--repeats','16','--batch-size','8','--temperatures','.3','.1','--output',str(run/'generation32')],'generation32')
  current=json.loads((run/'generation32/sampling.json').read_text());ids=set(json.loads((old/'training_ids.json').read_text()));comparisons={}
  for label,path in [('base',online/'base/sampling.json'),('positive8',old/'generation/sampling.json')]:
   baseline=json.loads(path.read_text());comparisons[label]={}
   for group in ['all32','original8','added24']:
    comparisons[label][group]={}
    for t in current:
     select=lambda rows:[r for r in rows if group=='all32' or ((r['sample_id'] in ids)==(group=='original8'))]
     b=select(baseline[t]['per_question']);c=select(current[t]['per_question'])
     comparisons[label][group][t]=dict(base_correct=sum(x['correct'] for x in b),current_correct=sum(x['correct'] for x in c),draws=sum(x['draws'] for x in c),**paired_bootstrap(dict(per_question=b),dict(per_question=c)))
  (run/'training_question_comparison.json').write_text(json.dumps(comparisons,indent=2))
  execute(['scripts/evaluate_grpo_positive_validation.py','--base',a.base,'--checkpoint',str(checkpoint),'--data',str(run/'validation128.jsonl'),'--output',str(run/'validation')],'validation')
  (run/'limitations.json').write_text(json.dumps(['Single seed; positive rollout distillation, not online GRPO.','Four times the successful-program exposure of fixed8 at the same per-program update count.','Validation deduplicated against local training JSONL; this does not establish absence from model pretraining.','Validation only at preselected step200; T=.3 primary, T=.1 secondary.'],indent=2))
  (run/'COMPLETE').touch();(run/'STAGE').write_text('complete\n')
 except BaseException:
  (run/'FAILED').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
