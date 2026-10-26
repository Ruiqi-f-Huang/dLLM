import json,ast,collections,random,hashlib
from pathlib import Path
out=Path('/root/autodl-tmp/s-flm/outputs/mathinstruct-audit')
rows=[json.loads(l) for l in (out/'verified-audit-candidates.jsonl').read_text().splitlines()]
def literal_return(code):
 f=ast.parse(code).body[0]; env={}
 def literal(v):
  if isinstance(v,ast.Constant):return True
  if isinstance(v,ast.UnaryOp):return literal(v.operand)
  if isinstance(v,ast.Name):return env.get(v.id,False)
  return False
 for n in f.body[:-1]:
  if isinstance(n,ast.Assign):
   flag=literal(n.value)
   for target in n.targets:
    if isinstance(target,ast.Name):env[target.id]=flag
  elif isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name):env[n.target.id]=False
  elif isinstance(n,(ast.If,ast.For)):
   for child in ast.walk(n):
    if isinstance(child,ast.Name) and isinstance(child.ctx,ast.Store):env[child.id]=False
 return literal(f.body[-1].value)

flagged=[r for r in rows if literal_return(r['code'])]
clean=[r for r in rows if not literal_return(r['code'])]
unique={}
for r in clean:
 key=(r['gsm8k_train_id'],ast.dump(ast.parse(r['code']),include_attributes=False))
 unique.setdefault(key,r)
groups=collections.defaultdict(list)
for r in unique.values():groups[r['gsm8k_train_id']].append(r)
best={qid:min(rs,key=lambda r:(r['tokens'],r['row_id'])) for qid,rs in groups.items()}
validation=set(random.Random(42).sample(range(7473),200))
train=[r for qid,r in sorted(best.items()) if qid not in validation]
summary={'strict_verified_rows':len(rows),'flag_direct_or_aliased_literal_return_rows':len(flagged),'after_literal_screen_rows':len(clean),'after_within_question_ast_dedup_rows':len(unique),'one_per_question_count':len(best),'proposed_validation_questions':200,'validation_questions_with_verified_program':len(set(best)&validation),'proposed_sft_train_questions':len(train),'proposed_sft_train_programs_if_all_distinct_variants':sum(len(v) for k,v in groups.items() if k not in validation),'seed':42,'split_method':'random.Random(42).sample(range(7473),200) before quality filtering; all variants grouped by original GSM8K train ID','proposed_train_length_min_median_max':None,'literal_flag_examples':flagged[:5]}
lengths=sorted(r['tokens'] for r in train)
summary['proposed_train_length_min_median_max']=[lengths[0],lengths[len(lengths)//2],lengths[-1]]
summary['sample_programs']=[best[q] for q in random.Random(42).sample(sorted(best),5)]
summary['validation_ids']=sorted(validation)
(out/'selection-feasibility.json').write_text(json.dumps(summary,indent=2))
(out/'audit-sampled-programs.json').write_text(json.dumps(summary['sample_programs'],indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ['sample_programs','validation_ids','literal_flag_examples']},indent=2))
print('Literal examples:',[(r['row_id'],r['question'],r['code']) for r in flagged[:3]])
for r in summary['sample_programs']:print('SAMPLE',r['gsm8k_train_id'],r['tokens'],r['code'])
