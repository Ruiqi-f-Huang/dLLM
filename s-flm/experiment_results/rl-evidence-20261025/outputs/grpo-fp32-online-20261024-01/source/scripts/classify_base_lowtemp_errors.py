"""Audit saved GSM8K generations with the existing evaluator; no model inference."""
import argparse
import ast
import collections
import contextlib
import csv
import hashlib
import io
import json
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sandbox_gsm8k import (
    _extract_code, _extract_gold_answer, _extract_text_answer, _numbers_equal,
    _safe_exec_no_timer, _time_limit, _to_number, evaluate_samples,
)


def audit(record, index):
    raw = record['response']
    # Same envelope extraction as the evaluator, before its syntax-prefix recovery.
    fence = re.search(r'```(?:python)?\s*(.*?)```', raw, re.DOTALL | re.IGNORECASE)
    strict = fence.group(1) if fence else raw
    start = strict.find('def ')
    if start != -1:
        strict = strict[start:]
    strict = strict.strip()
    code = _extract_code(raw)
    row = dict(index=index, question=record['prompt'], reference=record['response_ground_truth'],
               response=raw, extracted_code=code, gold=_extract_gold_answer(record['response_ground_truth']))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        try:
            compile(strict, '<strict>', 'exec')
            row['raw_parseable'] = True
        except (SyntaxError, ValueError, TypeError):
            row['raw_parseable'] = False
        try:
            compile(code, '<sample>', 'exec')
        except (SyntaxError, ValueError, TypeError) as exc:
            row.update(status='parse_error', error_type=type(exc).__name__, error=str(exc))
            return row
    try:
        with _time_limit(5.0):
            ns = _safe_exec_no_timer(code)
            fn = ns.get('simple_math_problem')
            if fn is None:
                row.update(status='missing_function')
                return row
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                out = fn()
            pred = _to_number(out)
            if pred is None:
                pred = _extract_text_answer(stdout.getvalue())
            row.update(return_repr=repr(out), prediction=pred)
            row['status'] = 'no_numeric_answer' if pred is None else ('correct' if _numbers_equal(pred, row['gold']) else 'wrong_numeric_answer')
    except Exception as exc:
        row.update(status='runtime_error', error_type=type(exc).__name__, error=str(exc))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--annotations', type=Path, help='Reviewed JSONL labels for every numeric-wrong case; no heuristic semantic classification.')
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    rows = [audit(record, i) for i, record in enumerate(data['records'])]
    counts = collections.Counter(r['status'] for r in rows)
    assert counts['correct'] == data['num_correct'] == 179
    for row, record in zip(rows, data['records']):
        assert (row['status'] == 'correct') == bool(evaluate_samples(record['response'], record['response_ground_truth'], 5.0))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'execution_audit.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    summary = dict(total=len(rows), status_counts=dict(counts),
                   raw_parseable=sum(r['raw_parseable'] for r in rows),
                   recovered_parseable=sum(r['status'] != 'parse_error' for r in rows),
                   runtime_error_types=dict(collections.Counter(r['error_type'] for r in rows if r['status'] == 'runtime_error')))
    strict_counts = collections.Counter('parse_error' if not r['raw_parseable'] else r['status'] for r in rows)
    summary.update(strict_status_counts=dict(strict_counts),
                   recovered_from_syntax_error=dict(collections.Counter(r['status'] for r in rows if not r['raw_parseable'])),
                   strict_runtime_error_types=dict(collections.Counter(r['error_type'] for r in rows if r['raw_parseable'] and r['status'] == 'runtime_error')),
                   input_sha256=hashlib.sha256(args.results.read_bytes()).hexdigest(),
                   protocol=dict(split='validation', n=632, model='Base EMA', temperature=0.1,
                                 nfe=32, sampler='ancestral', generation_seed=1, execution_timeout_seconds=5))
    (args.output / 'execution_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if args.annotations:
        annotations = [json.loads(line) for line in args.annotations.read_text().splitlines() if line.strip()]
        indexed = {r['index']: r for r in annotations}
        assert len(indexed) == len(annotations), 'Duplicate semantic labels'
        assert set(indexed) == {r['index'] for r in rows if r['status'] == 'wrong_numeric_answer'}, 'Missing or extraneous semantic labels'
        allowed = {'modeling', 'missing_steps', 'calculation', 'ambiguous_reference'}
        assert all(r['category'] in allowed and r['reason'] for r in annotations)
        for row in rows:
            if row['index'] in indexed:
                row.update(indexed[row['index']])
        semantic_counts = collections.Counter(r['category'] for r in annotations)
        semantic = dict(numeric_wrong=len(annotations), category_counts=dict(semantic_counts),
                        review='Single assistant semantic review of every numeric-wrong program against question and reference; not independent human adjudication.',
                        calculation_definition='Local arithmetic-expression implementation errors, including missing parentheses, duplicate factors/terms and loop count; not evidence of Python arithmetic malfunction.',
                        index_convention='Zero-based index in the saved 632-record results.json')
        (args.output / 'semantic_summary.json').write_text(json.dumps(semantic, ensure_ascii=False, indent=2) + '\n')
        (args.output / 'classified_records.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        with (args.output / 'numeric_wrong_classification.csv').open('w', newline='') as out:
            fields = ['index', 'category', 'reason', 'prediction', 'gold', 'question', 'extracted_code', 'reference']
            writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(r for r in rows if r['status'] == 'wrong_numeric_answer')
    compact = []
    for r in rows:
        if r['status'] != 'wrong_numeric_answer':
            continue
        tree = ast.parse(r['extracted_code'])
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
        compact.append(f"INDEX {r['index']} PRED {r['prediction']} GOLD {r['gold']}\nQ: {r['question']}\nREF: {r['reference']}\nCODE:\n{ast.unparse(tree)}\n")
    (args.output / 'wrong_numeric_review.txt').write_text('\n'.join(compact))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
