#!/usr/bin/env python3
"""Independent provenance, split, token and execution checks for positive128 data."""
import argparse,json,sys,hashlib,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 p=argparse.ArgumentParser()
 for n in ['run','source32','train-data','validation']:p.add_argument('--'+n,required=True)
 a=p.parse_args();run=Path(a.run)
 try:
  from grpo_fixed32 import read_jsonl
  from grpo_overfit import stable_seed
  from train_mdlm_grpo import execution_reward
  from hydra import compose,initialize_config_dir
  import dataloader
  while not (run/'programs128.jsonl').exists() or not (run/'questions128.jsonl').exists() or not (run/'collection_summary.json').exists():
   if (run/'FAILED').exists():raise RuntimeError('Collection pipeline failed: '+(run/'FAILED').read_text())
   time.sleep(10)
  programs=read_jsonl(run/'programs128.jsonl');questions=read_jsonl(run/'questions128.jsonl')
  train=read_jsonl(a.train_data);validation=read_jsonl(a.validation);source=Path(a.source32)
  original=read_jsonl(source/'questions.jsonl');pairs=read_jsonl(source/'pairs.jsonl')
  if len(programs)!=128 or len(questions)!=128:raise ValueError('Expected128 records.')
  qmap={r['sample_id']:r for r in questions};trainmap={r['sample_id']:r for r in train}
  if len(qmap)!=128 or {r['sample_id'] for r in programs}!=set(qmap):raise ValueError('Question/program identity mismatch.')
  norm=lambda s:' '.join(s.casefold().split())
  if len({norm(r['question']) for r in questions})!=128:raise ValueError('Duplicate question text.')
  if set(qmap)&{r['sample_id'] for r in validation} or {norm(r['question']) for r in questions}&{norm(r['question']) for r in validation}:raise ValueError('Validation leakage.')
  for sid,r in qmap.items():
   if sid not in trainmap or r['question']!=trainmap[sid]['question'] or str(r['answer'])!=str(trainmap[sid]['answer']):raise ValueError('Question/gold differs from training source.')
  oldids={r['sample_id'] for r in original};olds={r['sample_id']:r for r in pairs if r['reward']==1}
  if len(oldids)!=32 or not oldids<=set(qmap):raise ValueError('Original32 not retained.')
  candidates=read_jsonl(run/'candidate_rollouts.jsonl');groups={}
  for r in candidates:groups.setdefault(r['sample_id'],[]).append(r)
  order=json.loads((run/'candidate_order.json').read_text());actual=list(groups)
  if actual!=order[:len(actual)]:raise ValueError('Candidate order differs.')
  for sid,rows in groups.items():
   if len(rows)!=8 or [r['candidate'] for r in rows]!=list(range(8)):raise ValueError('Candidate budget not exactly8.')
   if any(r['seed']!=stable_seed('positive128-base-candidates',20261025,sid) for r in rows):raise ValueError('Unexpected candidate seed.')
  with initialize_config_dir(version_base=None,config_dir=str(ROOT/'configs')):cfg=compose(config_name='sft-a800-lora-control')
  tok=dataloader.get_tokenizer(cfg);sep=tok(str(cfg.data.separator),add_special_tokens=False).input_ids
  results=[]
  for index,r in enumerate(programs):
   sid=r['sample_id'];q=qmap[sid];ids=r['tokens'];prefix=[tok.bos_token_id]+tok(q['question'].strip(),add_special_tokens=False).input_ids+sep
   if len(ids)!=512 or r['prefix_length']!=len(prefix) or ids[:len(prefix)]!=prefix:raise ValueError('Token/prefix mismatch: '+sid)
   if any(not isinstance(i,int) or i<0 or i>=len(tok) for i in ids):raise ValueError('Invalid token id.')
   if r['reward']!=1 or r['execution'].get('infrastructure_failure'):raise ValueError('Invalid stored success.')
   if sid in oldids:
    if r['origin']!='retained32' or ids!=olds[sid]['tokens']:raise ValueError('Original success modified.')
   else:
    matching=[x for x in groups[sid] if x['reward']==1]
    if not matching or r['origin']!='new96' or r['candidate']!=matching[0]['candidate'] or ids!=matching[0]['tokens']:raise ValueError('Not first correct candidate.')
   suffix=ids[len(prefix):]
   if tok.eos_token_id in suffix:suffix=suffix[:suffix.index(tok.eos_token_id)]
   text=tok.decode(suffix,skip_special_tokens=True)
   if sid not in oldids and text!=r['text']:raise ValueError('Decoded text differs.')
   for attempt in range(3):
    detail=execution_reward(text,q['answer'],2.,details=True)
    if not detail.get('infrastructure_failure'):break
   if detail.get('infrastructure_failure') or detail['reward']!=1:raise ValueError('Independent execution failed: '+sid+' '+str(detail))
   results.append(dict(sample_id=sid,reward=detail['reward'],raw_correct=detail['raw']['correct'],execution=detail,retries=attempt))
   if (index+1)%16==0:print(json.dumps(dict(verified=index+1,total=128)),flush=True)
  summary=json.loads((run/'collection_summary.json').read_text())
  if summary['attempted_questions']!=len(groups) or summary['new96']!=96:raise ValueError('Collection summary mismatch.')
  hashes={n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in ['programs128.jsonl','questions128.jsonl','candidate_rollouts.jsonl','candidate_order.json']}
  (run/'data_verification.json').write_text(json.dumps(dict(status='passed',questions=128,retained32=32,new96=96,candidate_questions=len(groups),candidate_draws=len(candidates),independent_execution_successes=len(results),raw_correct=sum(r['raw_correct'] for r in results),validation_overlap=0,source_prefix_and_first_success_checks=True,hashes=hashes,per_program=results),indent=2))
  (run/'DATA_VERIFIED').touch();print('DATA_VERIFIED',flush=True)
 except BaseException:
  (run/'DATA_VERIFICATION_FAILED').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
