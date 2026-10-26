import json,sys,ast
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from sandbox_gsm8k import _safe_exec_no_timer,_to_number,_numbers_equal,_time_limit
root=Path('reports/base-lowtemp-error-analysis-20261023')
rows={r['index']:r for r in (json.loads(l) for l in (root/'execution_audit.jsonl').read_text().splitlines())}
patches={
9:[('2 - 0.5 / 0.25','2 - 0.5')],
40:[('(lions + rhinos) * rhinos * time_per_animal','(lions + rhinos) * time_per_animal')],
119:[('yards * yards * 2','yards * 2')],
125:[('range(3)','range(4)'),('        soup_weight /= 2\n        soup_weight /= 2','        soup_weight /= 2')],
231:[('20 / 2 / 3','20 / (2 / 3)')],
252:[('total_cost - change - change - table_cost - plates_cost','total_cost - change - table_cost - plates_cost')],
283:[('sheep_speed - sheepdog','sheepdog - sheep_speed')],
357:[('hours = total_time / 60','hours = total_time')],
367:[('beats_per_minute * 120 * hours_per_day','beats_per_minute * minutes_per_hour * hours_per_day')],
375:[('2 * 2 * (width + length)','2 * (width + length)')],
410:[('margo_donation - julie_donation','julie_donation - margo_donation')],
430:[('time_mary * 2 + time_susan','time_mary + time_susan')],
484:[('total_cost = shirt_cost + shirt_cost','total_cost = shirt_cost')],
567:[('apples + oranges + oranges','apples + oranges')],
627:[('amount_sold / 4 / 7','amount_sold / (4 / 7)')],
}
checks=[]
for i,edits in patches.items():
 code=ast.unparse(ast.parse(rows[i]['extracted_code']))
 for old,new in edits:
  assert old in code,(i,old)
  code=code.replace(old,new,1)
 with _time_limit(5):
  out=_safe_exec_no_timer(code)['simple_math_problem']()
 pred=_to_number(out)
 checks.append(dict(index=i,edits=edits,repaired_prediction=pred,gold=rows[i]['gold'],matches_gold=_numbers_equal(pred,rows[i]['gold'])))
assert all(r['matches_gold'] for r in checks)
(root/'calculation_local_repair_checks.json').write_text(json.dumps(dict(note='Post-hoc diagnostic edits only; original scores and generations unchanged. A local successful repair does not prove the latent reasoning cause or define an unbiased intervention.',checks=checks),ensure_ascii=False,indent=2)+'\n')
print('All',len(checks),'local calculation-expression repairs match reference answers.')
