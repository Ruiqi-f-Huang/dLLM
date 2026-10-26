"""Prepare local SFT/RL files from the pinned, executed MathInstruct audit."""

import argparse
import ast
import hashlib
import json
import random
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def literal_return(code):
    """Quarantine simple literal answers; this is not a reasoning verifier."""
    function = ast.parse(code).body[0]
    env = {}

    def literal(value):
        if isinstance(value, ast.Constant):
            return True
        if isinstance(value, ast.UnaryOp):
            return literal(value.operand)
        if isinstance(value, ast.Name):
            return env.get(value.id, False)
        return False

    for node in function.body[:-1]:
        if isinstance(node, ast.Assign):
            flag = literal(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    env[target.id] = flag
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            env[node.target.id] = False
        elif isinstance(node, (ast.If, ast.For)):
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    env[child.id] = False
    return literal(function.body[-1].value)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, default=root / 'data/mathinstruct-audit')
    parser.add_argument('--output-dir', type=Path, default=root / 'data/gsm8k_pot')
    args = parser.parse_args()
    audit, output = args.audit_dir, args.output_dir
    checks = json.loads((audit / 'artifact-hashes.json').read_text())
    for name, expected in checks.items():
        if sha256(audit / name) != expected:
            raise ValueError(f'Audit checksum mismatch: {name}')

    candidate_path = audit / 'verified-audit-candidates.jsonl'
    candidates = [json.loads(line) for line in candidate_path.read_text().splitlines()]
    execution = {
        r['row_id']: r
        for r in map(json.loads, (audit / 'execution-audit.jsonl').read_text().splitlines())
    }
    best = {}
    flagged = 0
    for row in candidates:
        if row['source'] != 'data/PoT/gsm_gpt4.json' or row['tokens'] > 512:
            raise ValueError('Unexpected source or length')
        if execution[row['row_id']]['status'] != 'verified':
            raise ValueError('Candidate lacks execution verification')
        if literal_return(row['code']):
            flagged += 1
            continue
        qid = row['gsm8k_train_id']
        previous = best.get(qid)
        if previous is None or (row['tokens'], row['row_id']) < (previous['tokens'], previous['row_id']):
            best[qid] = row

    expected = json.loads((audit / 'selection-feasibility.json').read_text())
    if len(best) != expected['one_per_question_count']:
        raise ValueError('Selected question count differs from audited count')
    question_ids = sorted(best)
    validation_count = (len(question_ids) + 9) // 10
    validation_ids = set(random.Random(42).sample(question_ids, validation_count))
    train_ids = set(question_ids) - validation_ids
    assert not train_ids & validation_ids
    output.mkdir(parents=True, exist_ok=True)
    written = {}
    for split, ids in [('train', train_ids), ('validation', validation_ids)]:
        sft, rl = [], []
        for qid in sorted(ids):
            row = best[qid]
            answer = execution[row['row_id']]['gold_answer']
            common = {
                'sample_id': f'gsm8k_train_{qid:04d}',
                'gsm8k_train_id': qid,
                'question': row['question'],
                'answer': str(answer),
                'reference_answer': row['gold'],
                'mathinstruct_row_id': row['row_id'],
                'source': row['source'],
            }
            sft.append({**common, 'code': row['code'], 'unpadded_total_tokens': row['tokens']})
            # No reference program in the RL records: rewards use numeric gold.
            rl.append(common)
        for kind, records in [('sft', sft), ('rl', rl)]:
            path = output / f'{kind}_{split}.jsonl'
            # Escape Unicode line separators too, so splitlines-based readers
            # cannot accidentally split a single JSON record into several rows.
            path.write_text(''.join(json.dumps(r, ensure_ascii=True) + '\n' for r in records))
            written[path.name] = {'rows': len(records), 'sha256': sha256(path)}

    manifest = {
        'seed': 42,
        'split_method': 'sample ceil(10% of sorted eligible GSM8K Train IDs) with random.Random(42)',
        'one_program_per_question': True,
        'program_selection': 'minimum (unpadded total tokens, original MathInstruct row ID)',
        'candidate_rows': len(candidates),
        'literal_return_quarantined_rows': flagged,
        'eligible_questions': len(best),
        'train_ids': sorted(train_ids),
        'validation_ids': sorted(validation_ids),
        'candidate_sha256': sha256(candidate_path),
        'upstream_sources': json.loads((audit / 'sources.json').read_text()),
        'files': written,
        'test': 'Original GSM8K Test, excluded from all training files.',
        'status': 'Mechanically verified preparation; manual semantic quality review and SFT pilot still pending.',
        'compatibility': 'Use checkpoint tokenizer, literal backslash+n separator, length 512, answer-only mask, and padding supervision.',
    }
    (output / 'split_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'eligible_questions': len(best), 'files': written}, indent=2))


if __name__ == '__main__':
    main()
