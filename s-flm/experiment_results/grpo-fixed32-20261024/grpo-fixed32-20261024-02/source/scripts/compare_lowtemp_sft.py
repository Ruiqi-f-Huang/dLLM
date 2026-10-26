"""Paired correctness and code changes, all models under one decoding protocol."""
import argparse
import ast
import json
import math
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from sandbox_gsm8k import _extract_code, evaluate_samples
warnings.filterwarnings('ignore', category=SyntaxWarning)

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
root = parser.parse_args().root
labels = ['base', 'uniform200', 'weighted200', 'weighted711']
models = {}

def code_ast(response):
    try:
        code = _extract_code(response)
        compile(code, '<sample>', 'exec')
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
        return ast.dump(tree, include_attributes=False)
    except (SyntaxError, ValueError, TypeError):
        return None

for label in labels:
    config = OmegaConf.load(root / label / '.hydra/config.yaml')
    assert config.sampler.temperature == .1 and config.sampler.steps == 32
    assert config.sampler.predictor == 'ancestral'
    assert config.model.length == 512 and config.seed == 1
    assert config.loader.eval_batch_size == 32 and not config.eval.disable_ema
    assert config.lora.enabled == (label != 'base')
    data = json.loads((root / label / 'results.json').read_text())
    records = data['records']
    flags = [bool(evaluate_samples(r['response'], r['response_ground_truth'], 5.)) for r in records]
    assert len(records) == 632 and sum(flags) == data['num_correct']
    models[label] = dict(records=records, flags=flags,
                         ast=[code_ast(r['response']) for r in records])

base = models['base']
rows = []
for label, model in models.items():
    assert [(r['prompt'], r['response_ground_truth']) for r in model['records']] == [(r['prompt'], r['response_ground_truth']) for r in base['records']]
    gained = sum(not a and b for a, b in zip(base['flags'], model['flags']))
    lost = sum(a and not b for a, b in zip(base['flags'], model['flags']))
    discordant = gained + lost
    # Exact two-sided paired binomial (McNemar) test; single-seed exploratory.
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(gained, lost) + 1)) / 2 ** discordant) if discordant else 1.
    rows.append(dict(model=label, correct=sum(model['flags']), total=632,
                     accuracy=sum(model['flags']) / 632,
                     gained_from_base=gained, lost_from_base=lost,
                     kept_base_correct=sum(a and b for a,b in zip(base['flags'],model['flags'])),
                     parseable=sum(a is not None for a in model['ast']),
                     unclosed_quotes=sum(bool(r['response'].count('"""')%2 or r['response'].count("'''")%2) for r in model['records']),
                     changed_outputs=sum(a['response'] != b['response'] for a,b in zip(base['records'],model['records'])),
                     changed_ast_excluding_docstrings=sum(a != b for a,b in zip(base['ast'],model['ast']) if a is not None and b is not None),
                     both_parseable=sum(a is not None and b is not None for a,b in zip(base['ast'],model['ast'])),
                     paired_exact_p_vs_base=p))
summary = dict(protocol=dict(split='validation', n=632, temperature=.1, sampler='ancestral',
                             nfe=32, weights='EMA', seed=1, batch_size=32, length=512),
               rows=rows, limitations=['Single generation seed; exploratory comparisons, no multiplicity correction.',
                                      'AST differences do not establish mathematical improvement.'])
(root/'comparison.json').write_text(json.dumps(summary, indent=2))
with (root/'paired_by_question.jsonl').open('w') as out:
    for i, record in enumerate(base['records']):
        out.write(json.dumps(dict(index=i, question=record['prompt'], correct={label: models[label]['flags'][i] for label in labels}),ensure_ascii=False)+'\n')
print(json.dumps(summary, indent=2))
