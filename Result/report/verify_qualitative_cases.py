"""Re-execute selected archived programs; no model generation or new training."""
import json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
F=ROOT/'Result/final-test-20261026-01'
DATA=ROOT/'Result/data/qualitative_cases.json'
worker="""import json,sys,ast
sys.path.insert(0,sys.argv[1])
import sandbox_gsm8k as s
text=json.load(sys.stdin)['text']
try:
 ast.parse(text)
 with s._time_limit(2.):
  ns=s._safe_exec_no_timer(text);v=ns['simple_math_problem']()
 print(json.dumps(dict(status='returned',value=v)))
except Exception as e:print(json.dumps(dict(status='error',error=type(e).__name__,message=str(e))))
"""
cases=[]
for index,kind,models in [(134,'wrong-to-right',['base_ema','positive32_raw']),(190,'sft-failure',['base_ema','sft711_ema']),(1143,'right-to-wrong',['base_ema','positive32_raw'])]:
 case=dict(index=index,sample_id=f'gsm8k_test_{index:04d}',seed=1,kind=kind,models={})
 for m in models:
  records=[json.loads(x) for x in (F/f'{m}-seed1.jsonl').read_text().splitlines()];r=records[index]
  result=subprocess.run([sys.executable,'-c',worker,str(ROOT/'s-flm')],input=json.dumps({'text':r['text']}),capture_output=True,text=True,timeout=6,check=True)
  case['question']=r['question'];case['gold_answer']=r['answer'];case['models'][m]=dict(text=r['text'],archived_reward=r['reward'],raw_reexecution=json.loads(result.stdout))
 cases.append(case)
DATA.write_text(json.dumps(cases,indent=2,ensure_ascii=False))
for c in cases:print(c['sample_id'],{m:r['raw_reexecution'] for m,r in c['models'].items()})
