import json,re,ast,collections,unicodedata
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from omegaconf import OmegaConf
out=Path('/root/autodl-tmp/s-flm/outputs/mathinstruct-audit'); m=json.loads((out/'sources.json').read_text())
def path(repo,f):return m[repo]['files'][f]['path']
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).split())
gsm={s:pq.read_table(path('openai/gsm8k',f'main/{s}-00000-of-00001.parquet')).to_pylist() for s in ['train','test']}
index={s:collections.defaultdict(list) for s in gsm}
for s,rows in gsm.items():
 for i,r in enumerate(rows):index[s][norm(r['question'])].append(i)
lengths={s:sorted({len(q) for q in index[s]},reverse=True) for s in gsm}
def match(text,split):
 n=norm(text)
 for z in lengths[split]:
  if len(n)>=z and n[:z] in index[split] and (len(n)==z or n[z]==' '):return index[split][n[:z]],n[z:].strip()
 return [],None
rows=json.loads(Path(path('TIGER-Lab/MathInstruct','MathInstruct.json')).read_text())
stats={'mathinstruct_rows':len(rows),'source_counts':dict(collections.Counter(r['source'] for r in rows)),'gsm8k_rows':{k:len(v) for k,v in gsm.items()}}
pot=[(i,r) for i,r in enumerate(rows) if r['source']=='data/PoT/gsm_gpt4.json']
counts=collections.Counter(); suffixes=collections.Counter(); imports=collections.Counter(); shapes=collections.Counter(); examples={}; candidates=[]
covered=set(); testcovered=set()
manifest=json.loads(Path('/root/autodl-tmp/s-flm/outputs/stage0-step1/download-manifest.json').read_text())
tok=AutoTokenizer.from_pretrained(manifest['tokenizer']['local_path'],local_files_only=True)
config=OmegaConf.load('/root/autodl-tmp/s-flm/outputs/stage0-step1/load-config.yaml')
sep=tok(config.data.separator,add_special_tokens=False).input_ids
safe_calls={'abs','min','max','sum','len','range','enumerate','int','float','str','bool','round'}
def length(q,code):return 2+len(tok(q.strip(),add_special_tokens=False).input_ids)+len(sep)+len(tok(code.strip(),add_special_tokens=False).input_ids)
def wrap(body,q=None):
 if q is not None:body=[ast.Expr(ast.Constant(q))]+body
 fn=ast.FunctionDef(name='simple_math_problem',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[],returns=ast.Name(id='int',ctx=ast.Load()))
 return ast.unparse(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])))+'\n'
for rowid,r in pot:
 ids,suffix=match(r['instruction'],'train'); testids,_=match(r['instruction'],'test')
 if testids:counts['test_prefix_match_rows']+=1;testcovered.update(testids)
 if not ids:counts['not_train_prefix_match_rows']+=1;examples.setdefault('unmatched',r);continue
 counts['train_prefix_match_rows']+=1;covered.update(ids);suffixes[suffix]+=1
 if len(ids)!=1:counts['ambiguous_train_match_rows']+=1;continue
 qid=ids[0]; q=gsm['train'][qid]['question'];gold=gsm['train'][qid]['answer']
 code=r['output'].strip(); fence=re.fullmatch(r'```(?:python)?\s*\n?(.*?)\n?```',code,re.S|re.I)
 if fence:code=fence.group(1).strip();counts['whole_fence_removed']+=1
 try:tree=ast.parse(code)
 except SyntaxError:counts['syntax_error']+=1;examples.setdefault('syntax_error',r);continue
 counts['syntax_ok']+=1
 imp=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
 for n in imp:
  if isinstance(n,ast.Import):imports.update(a.name for a in n.names)
  else:imports[n.module or '']+=1
 if imp:counts['has_import']+=1;examples.setdefault('import',r)
 calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
 prints=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='print']
 shapes[f'print_calls_{len(prints)}']+=1
 functions=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
 if functions:counts['has_function_or_class']+=1;examples.setdefault('function',r)
 last=tree.body[-1] if tree.body else None
 lastprint=isinstance(last,ast.Expr) and isinstance(last.value,ast.Call) and isinstance(last.value.func,ast.Name) and last.value.func.id=='print'
 if lastprint:counts['last_statement_print']+=1
 reason=None
 if testids:reason='test_overlap'
 elif imp:reason='import'
 elif functions:reason='function_or_class'
 elif len(prints)!=1 or not lastprint or len(last.value.args)!=1 or last.value.keywords or isinstance(last.value.args[0],ast.Starred):reason='noncanonical_print'
 elif any(isinstance(n,(ast.Attribute,ast.Global,ast.Nonlocal,ast.Lambda,ast.While,ast.With,ast.Try,ast.Raise,ast.Delete)) for n in ast.walk(tree)):reason='complex_ast'
 elif any(not isinstance(n.func,ast.Name) or n.func.id not in safe_calls|{'print'} for n in calls):reason='unsupported_call'
 elif any(isinstance(n,ast.Name) and n.id.startswith('__') for n in ast.walk(tree)):reason='dunder_name'
 if reason:counts['reject_'+reason]+=1;examples.setdefault('reject_'+reason,r);continue
 body=tree.body[:-1]+[ast.Return(last.value.args[0])]
 wrapped=wrap(body,q); compact=wrap(body)
 candidates.append({'row_id':rowid,'gsm8k_train_id':qid,'question':q,'gold':gold,'source':r['source'],'instruction_suffix':suffix,'original_code':code,'code':wrapped,'compact_code':compact,'tokens':length(q,wrapped),'compact_tokens':length(q,compact)})
counts['conservative_convertible']=len(candidates)
stats.update({'pot_rows':len(pot),'counts':dict(counts),'matched_train_unique_questions':len(covered),'matched_test_unique_questions':len(testcovered),'instruction_suffix_counts':dict(suffixes),'imports':dict(imports),'shapes':dict(shapes),'candidate_length':{'with_docstring_le512':sum(x['tokens']<=512 for x in candidates),'compact_le512':sum(x['compact_tokens']<=512 for x in candidates)},'examples':examples})
tiny=next(pq.ParquetFile(path('TinyGSM/TinyGSM','data/train-00000-of-00017.parquet')).iter_batches(batch_size=1000)).to_pylist()
tc=collections.Counter(); tl=[]
for r in tiny:
 try:
  t=ast.parse(r['code']); fs=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='simple_math_problem']
  tc['parseable']+=1
  if fs:
   tc['target_function']+=1
   if ast.get_docstring(fs[0]):tc['with_docstring']+=1
   if any(isinstance(n,ast.Return) for n in ast.walk(fs[0])):tc['with_return']+=1
 except SyntaxError:tc['syntax_error']+=1
 tl.append(length(r['question'],r['code']))
stats['tinygsm_first1000']={'sample_size':len(tiny),'scope':'first 1000 rows of shard 0; not random or full dataset','counts':dict(tc),'total_tokens_le512':sum(n<=512 for n in tl),'examples':tiny[:3]}
(out/'static-audit.json').write_text(json.dumps(stats,indent=2))
(out/'conservative-candidates.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in candidates))
print(json.dumps({k:v for k,v in stats.items() if k not in ['examples','tinygsm_first1000']},indent=2),flush=True)
print('Tiny sample:',stats['tinygsm_first1000']['counts'],stats['tinygsm_first1000']['total_tokens_le512'],flush=True)
print('STATIC AUDIT COMPLETE',flush=True)
