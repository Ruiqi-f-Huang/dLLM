"""Filter Base candidates and construct two SFT sets differing only in target code."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import random
import re
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
import dataloader
from sandbox_gsm8k import _extract_code, evaluate_samples
warnings.filterwarnings('ignore', category=SyntaxWarning)

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
parser.add_argument('--exclude-file', type=Path)
args = parser.parse_args()
run = args.run
out = ROOT/'data/gsm8k_pot_selftrain_pilot'
pool = [json.loads(l) for l in (out/'generation_pool.jsonl').read_text().splitlines() if l.strip()]
valid = [json.loads(l) for l in (ROOT/'data/gsm8k_pot/sft_validation.jsonl').read_text().splitlines() if l.strip()]
assert len(pool) == 512
assert not {r['sample_id'] for r in pool} & {r['sample_id'] for r in valid}
tok = dataloader.get_tokenizer(OmegaConf.create({'data': {'tokenizer_name_or_path':'HuggingFaceTB/SmolLM-135M','tokenizer_revision':'1d461723eec654e65efdc40cf49301c89c0c92f4'}}))
sep = tok('\\n', add_special_tokens=False).input_ids

def load(label, temperature, seed):
    c = OmegaConf.load(run/'acquisition'/label/'.hydra/config.yaml')
    assert c.sampler.temperature == temperature and c.seed == seed
    assert c.sampler.steps == 32 and not c.eval.disable_ema and not c.lora.enabled
    data = json.loads((run/'acquisition'/label/'results.json').read_text())
    assert len(data['records']) == len(pool)
    for record, source in zip(data['records'], pool):
        assert record['prompt'] == source['question']
        assert record['response_ground_truth'] == source['reference_answer']
    flags = [bool(evaluate_samples(r['response'],r['response_ground_truth'],5.)) for r in data['records']]
    assert sum(flags) == data['num_correct']
    return data['records'], flags

low_records, low_flags = load('lowtemp', .1, 1)
normalize = lambda text: re.sub(r'\s+', ' ', text).strip()
rejections = Counter()
candidates = []
by_question = [[] for _ in pool]
raw_pass_counts = [0]*len(pool)
excluded = json.loads(args.exclude_file.read_text()) if args.exclude_file else {}
for seed in [11,22,33,44]:
    records, flags = load(f'candidate{seed}', .7, seed)
    for index, (record, source, passed) in enumerate(zip(records,pool,flags)):
        status = dict(sample_id=source['sample_id'],seed=seed,raw_correct=passed)
        reason = None
        if not passed:
            reason = 'wrong_execution'
        else:
            raw_pass_counts[index] += 1
            code = _extract_code(record['response']).strip()
            try:
                compile(code,'<candidate>','exec')
                tree = ast.parse(code)
                functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='simple_math_problem']
                if len(functions) != 1:
                    reason = 'function_structure'
                elif normalize(ast.get_docstring(functions[0]) or '') != normalize(source['question']):
                    reason = 'docstring_not_exact_question'
                elif not any(isinstance(n,(ast.BinOp,ast.AugAssign)) or (isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'sum','min','max','abs','len'}) for n in ast.walk(functions[0])):
                    reason = 'no_arithmetic_hardcode_guard'
                else:
                    prefix_len = 1+len(tok(source['question'].strip(),add_special_tokens=False).input_ids)+len(sep)
                    length = prefix_len+len(tok(code,add_special_tokens=False).input_ids)+1
                    if length > 512:
                        reason = 'over_512_no_truncation'
                    elif not evaluate_samples(code,source['reference_answer'],5.):
                        reason = 'extraction_changed_execution'
                    else:
                        status.update(code=code,sequence_length=length,extracted_changed=code!=record['response'].strip())
            except (SyntaxError,ValueError,TypeError):
                reason = 'invalid_extracted_code'
        if source['sample_id'] in excluded:
            reason = 'manual_exclusion'
        status['accepted'] = reason is None
        status['reason'] = reason
        if reason:
            rejections[reason] += 1
        else:
            by_question[index].append(status)
        candidates.append(status)

original, generated, selected = [], [], []
for index, (source, options) in enumerate(zip(pool,by_question)):
    if not options:
        continue
    # First accepted seed; no validation-based target selection or shortest-code preference.
    option = options[0]
    common = {k:source[k] for k in ['sample_id','gsm8k_train_id','question','answer','reference_answer']}
    original.append(dict(common,code=source['code']))
    generated.append(dict(common,code=option['code']))
    selected.append(dict(sample_id=source['sample_id'],pool_index=index,lowtemp_correct=low_flags[index],selected_seed=option['seed'],accepted_candidates=len(options),raw_correct_candidates=raw_pass_counts[index],sequence_length=option['sequence_length'],extracted_changed=option['extracted_changed']))
assert original and len(original) == len(generated)
for a,b in zip(original,generated):
    assert {k:v for k,v in a.items() if k!='code'} == {k:v for k,v in b.items() if k!='code'}
for label, rows in [('original',original),('generated',generated)]:
    (out/f'sft_train_{label}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in rows))
def normalized_program(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
            node.body = node.body[1:]
    return ast.dump(tree,include_attributes=False)

summary = dict(pool_questions=512,candidates=2048,candidate_temperature=.7,candidate_seeds=[11,22,33,44],lowtemp_correct=sum(low_flags),raw_correct_candidates=sum(raw_pass_counts),raw_correct_candidate_rate=sum(raw_pass_counts)/2048,raw_covered_questions=sum(n>0 for n in raw_pass_counts),accepted_candidates=sum(len(x) for x in by_question),accepted_questions=len(selected),accepted_question_coverage=len(selected)/512,selected_lowtemp_correct=sum(x['lowtemp_correct'] for x in selected),selected_lowtemp_wrong=sum(not x['lowtemp_correct'] for x in selected),rejections=dict(rejections),manual_exclusions=excluded,training_eligible=len(selected)>=32 and sum(not x['lowtemp_correct'] for x in selected)>=8)
summary.update(selected_targets_exactly_original=sum(a['code'].strip()==b['code'].strip() for a,b in zip(original,generated)), selected_targets_same_ast_excluding_docstrings=sum(normalized_program(a['code'])==normalized_program(b['code']) for a,b in zip(original,generated)), original_mean_code_tokens=sum(len(tok(x['code'].strip(),add_special_tokens=False).input_ids) for x in original)/len(original), generated_mean_code_tokens=sum(len(tok(x['code'].strip(),add_special_tokens=False).input_ids) for x in generated)/len(generated))
run.joinpath('filter_summary.json').write_text(json.dumps(summary,indent=2))
run.joinpath('candidate_filter.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in candidates))
run.joinpath('selected_questions.json').write_text(json.dumps(selected,indent=2))
manifest = {'summary':summary,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')}}
out.joinpath('training_manifest.json').write_text(json.dumps(manifest,indent=2))
# A fixed review sample, stratified by success of the low-temperature baseline.
review = []
rng = random.Random(20261023)
for flag in [False, True]:
    indices = [i for i,x in enumerate(selected) if x['lowtemp_correct']==flag]
    review += rng.sample(indices,min(16,len(indices)))
remaining = [i for i in range(len(selected)) if i not in review]
review += rng.sample(remaining,min(32-len(review),len(remaining)))
review_rows = [dict(metadata=selected[i],question=generated[i]['question'],answer=generated[i]['answer'],code=generated[i]['code'],original_code=original[i]['code']) for i in review]
run.joinpath('review32.json').write_text(json.dumps(review_rows,ensure_ascii=False,indent=2))
print(json.dumps(summary,indent=2))
