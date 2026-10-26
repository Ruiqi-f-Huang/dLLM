import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,sys,ast,textwrap,io,contextlib,resource,collections,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,'/root/s-flm')
import sandbox_gsm8k as s
out=Path('/root/autodl-tmp/s-flm/outputs/mathinstruct-audit')
def init_worker():
 resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
def check(r):
 result={'row_id':r['row_id'],'gsm8k_train_id':r['gsm8k_train_id'],'tokens':r['tokens']}
 try:
  with s._time_limit(1):
   raw='def audit_original():\n'+textwrap.indent(r['original_code'],'    ')
   ns=s._safe_exec_no_timer(raw); buf=io.StringIO()
   with contextlib.redirect_stdout(buf),contextlib.redirect_stderr(io.StringIO()):ns['audit_original']()
   original=s._extract_text_answer(buf.getvalue())
   ns=s._safe_exec_no_timer(r['code']); value=ns['simple_math_problem']()
   predicted=s._to_number(value);gold=s._extract_gold_answer(r['gold'])
   result.update({'original_answer':original,'predicted_answer':predicted,'gold_answer':gold,'numeric_return':isinstance(value,(int,float)) and not isinstance(value,bool),'original_matches_gold':bool(s._numbers_equal(original,gold)),'converted_matches_gold':bool(s._numbers_equal(predicted,gold)),'conversion_equivalent':bool(s._numbers_equal(original,predicted))})
   result['status']='verified' if all(result[k] for k in ['numeric_return','original_matches_gold','converted_matches_gold','conversion_equivalent']) else 'verification_failed'
 except Exception as e:result.update({'status':'execution_error','error_type':type(e).__name__,'error':str(e)[:200]})
 return result
if __name__=='__main__':
 start=time.perf_counter()
 candidates=[json.loads(l) for l in (out/'conservative-candidates.jsonl').read_text().splitlines()]
 results=[]
 with ProcessPoolExecutor(max_workers=4,initializer=init_worker) as pool:
  for i,r in enumerate(pool.map(check,candidates,chunksize=64)):
   results.append(r)
   if (i+1)%2000==0:print('Executed',i+1,flush=True)
 (out/'execution-audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
 lookup={r['row_id']:r for r in results}
 accepted=[r for r in candidates if lookup[r['row_id']]['status']=='verified' and r['tokens']<=512]
 summary={'execution_rows':len(results),'status_counts':dict(collections.Counter(r['status'] for r in results)),'error_counts':dict(collections.Counter(r.get('error_type') for r in results if r['status']=='execution_error')),'verified_original_matches_gold':sum(r.get('original_matches_gold',False) for r in results),'verified_converted_matches_gold':sum(r.get('converted_matches_gold',False) for r in results),'conversion_equivalent':sum(r.get('conversion_equivalent',False) for r in results),'strict_verified_with_docstring_le512_rows':len(accepted),'strict_verified_unique_questions':len(set(r['gsm8k_train_id'] for r in accepted)),'elapsed_seconds':time.perf_counter()-start,'scope':'All conservative candidates; 4 CPU processes, 1-second per-pair timeout, 1GiB address-space limit, existing restricted builtins; no GPU/training.'}
 (out/'execution-summary.json').write_text(json.dumps(summary,indent=2))
 (out/'verified-audit-candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in accepted))
 print(json.dumps(summary,indent=2),flush=True)
 print('EXECUTION AUDIT COMPLETE',flush=True)
