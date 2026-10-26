"""Matched target-source SFT pilot: per-seed validation and paired transitions."""
import argparse
import ast
import json
import math
import statistics
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from sandbox_gsm8k import _extract_code, evaluate_samples
warnings.filterwarnings('ignore',category=SyntaxWarning)

p=argparse.ArgumentParser();p.add_argument('run',type=Path);run=p.parse_args().run
models={}

def normalized_ast(response):
    try:
        code=_extract_code(response);compile(code,'<sample>','exec');tree=ast.parse(code)
        for n in ast.walk(tree):
            if isinstance(n,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.body and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant) and isinstance(n.body[0].value.value,str):n.body=n.body[1:]
        return ast.dump(tree,include_attributes=False)
    except (SyntaxError,ValueError,TypeError):return None

for seed in [1,2]:
    for label in ['base','original','generated']:
        directory=run/'evaluation'/f'{label}-seed{seed}'
        c=OmegaConf.load(directory/'.hydra/config.yaml')
        assert c.seed==seed and c.sampler.temperature==.1 and c.sampler.steps==32
        assert c.sampler.predictor=='ancestral' and c.model.length==512 and c.loader.eval_batch_size==32
        assert not c.eval.disable_ema and c.lora.enabled==(label!='base')
        d=json.loads((directory/'results.json').read_text());rs=d['records']
        flags=[bool(evaluate_samples(r['response'],r['response_ground_truth'],5.)) for r in rs]
        assert len(rs)==632 and sum(flags)==d['num_correct']
        models[label,seed]={'records':rs,'flags':flags,'ast':[normalized_ast(r['response']) for r in rs]}

def compare(reference,candidate):
    assert [(r['prompt'],r['response_ground_truth']) for r in reference['records']]==[(r['prompt'],r['response_ground_truth']) for r in candidate['records']]
    gained=sum(not a and b for a,b in zip(reference['flags'],candidate['flags']))
    lost=sum(a and not b for a,b in zip(reference['flags'],candidate['flags']))
    n=gained+lost
    exact=min(1.,2*sum(math.comb(n,k) for k in range(min(gained,lost)+1))/2**n) if n else 1.
    return {'reference_correct':sum(reference['flags']),'candidate_correct':sum(candidate['flags']),'gained':gained,'lost':lost,'paired_exact_p':exact,'changed_outputs':sum(a['response']!=b['response'] for a,b in zip(reference['records'],candidate['records'])),'changed_ast_excluding_docstrings':sum(a!=b for a,b in zip(reference['ast'],candidate['ast']) if a is not None and b is not None),'both_parseable':sum(a is not None and b is not None for a,b in zip(reference['ast'],candidate['ast']))}

rows=[]
for seed in [1,2]:
    for label in ['base','original','generated']:
        m=models[label,seed]
        rows.append({'model':label,'seed':seed,'correct':sum(m['flags']),'total':632,'accuracy':sum(m['flags'])/632,'parseable':sum(a is not None for a in m['ast']),'unclosed_quotes':sum(bool(r['response'].count('"""')%2 or r['response'].count("'''")%2) for r in m['records']),'vs_base':compare(models['base',seed],m),'vs_original':compare(models['original',seed],m)})
means={label:{'correct_by_seed':[sum(models[label,s]['flags']) for s in [1,2]],'mean_correct':statistics.mean(sum(models[label,s]['flags']) for s in [1,2]),'mean_accuracy':statistics.mean(sum(models[label,s]['flags'])/632 for s in [1,2])} for label in ['base','original','generated']}
training={label:json.loads((run/f'train-{label}/metrics/version_0/summary.json').read_text()) for label in ['original','generated']}
assert all(x['final_step']==200 and x['successful_updates_session']==200 and x['amp_skipped_updates_session']==0 and x['nonfinite_gradient_updates_session']==0 for x in training.values())
assert training['original']['examples_seen_session']==training['generated']['examples_seen_session']
summary={'protocol':{'split':'validation','questions':632,'seeds':[1,2],'temperature':.1,'nfe':32,'sampler':'ancestral','weights':'EMA'},'data':json.loads((run/'filter_summary.json').read_text()),'training':training,'means':means,'rows':rows,'base_seed1_reused_from':'lowtemp-sft-comparison-20261023-01/base','limitations':['Single training seed; two generation seeds.','Scores on repeated validation questions across seeds are not independent; significance reported separately per seed.','Self-training targets use model-generated programs filtered by correctness, exact question docstring and length; the contrast tests the resulting target-source package.']}
(run/'comparison.json').write_text(json.dumps(summary,indent=2))
with (run/'validation_paired.jsonl').open('w') as out:
    for i,r in enumerate(models['base',1]['records']):out.write(json.dumps({'index':i,'question':r['prompt'],'correct':{f'{label}-seed{s}':models[label,s]['flags'][i] for label in ['base','original','generated'] for s in [1,2]}},ensure_ascii=True)+'\n')
print(json.dumps(summary,indent=2))
