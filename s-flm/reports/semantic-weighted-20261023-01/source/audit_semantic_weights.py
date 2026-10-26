"""Audit all BPE labels and save 32 deterministic examples before training."""
import argparse
import ast
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from semantic_weights import code_weights, critical_spans

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-135M', revision='1d461723eec654e65efdc40cf49301c89c0c92f4', local_files_only=True)
counts = {}
examples = []
for split in ['train', 'validation']:
    records = [json.loads(x) for x in Path(f'data/gsm8k_pot/sft_{split}.jsonl').read_text().splitlines() if x.strip()]
    critical = total = supervised = 0
    selected = set(random.Random(1).sample(range(len(records)), 32))
    for i, row in enumerate(records):
        code = row['code'].strip()
        ids, weights = code_weights(code, tok, 2)
        assert ids == tok(code, add_special_tokens=False).input_ids
        spans = critical_spans(code)
        assert all(0 <= a < b <= len(code) for a, b in spans)
        # Every STRING token (including multiline question docstrings) is
        # excluded unless part of a return expression by the explicit policy.
        assert set(weights) <= {1., 2.}
        critical += weights.count(2.)
        total += len(weights)
        prefix = 1 + len(tok(row['question'].strip(), add_special_tokens=False).input_ids) + len(tok('\\n', add_special_tokens=False).input_ids)
        supervised += 512 - prefix
        if i in selected:
            examples.append(dict(split=split, sample_id=row['sample_id'], code=code,
                                 highlighted=[code[a:b] for a, b in spans],
                                 pieces=[{'text': code[a:b], 'weight': w} for (a,b), w in zip(tok(code, add_special_tokens=False, return_offsets_mapping=True).offset_mapping, weights)]))
    counts[split] = dict(samples=len(records), code_tokens=total, critical_bpe_tokens=critical,
                         supervised_tokens=supervised, critical_fraction=critical/supervised,
                         weighted_mass=supervised+critical)
(args.output/'audit_summary.json').write_text(json.dumps(counts, indent=2))
(args.output/'audit_examples.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False))
print(json.dumps(counts, indent=2))
