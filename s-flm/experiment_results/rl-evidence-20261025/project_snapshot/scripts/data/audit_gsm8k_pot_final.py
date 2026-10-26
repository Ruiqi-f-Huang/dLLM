"""Re-audit the frozen GSM8K PoT splits without rewriting their programs."""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
import re
import signal
import warnings
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SAFE_CALLS = {
    "abs", "min", "max", "sum", "len", "range", "enumerate",
    "int", "float", "str", "bool", "round",
}
FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.AsyncFunctionDef, ast.ClassDef,
    ast.Attribute, ast.Global, ast.Nonlocal, ast.Lambda, ast.While,
    ast.With, ast.AsyncWith, ast.Try, ast.Raise, ast.Delete,
)


class ExecutionTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    del signum, frame
    raise ExecutionTimeout()


@contextlib.contextmanager
def time_limit(seconds):
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_number(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(round(value)) if abs(value - round(value)) < 1e-6 else value
    if isinstance(value, str):
        match = re.search(r"[-+]?\$?\d[\d,]*\.?\d*", value)
        if match is None:
            return None
        text = match.group(0).replace(",", "").replace("$", "")
        return to_number(float(text)) if "." in text else int(text)
    return None


def numbers_equal(left, right):
    if left is None or right is None:
        return False
    if isinstance(left, float) or isinstance(right, float):
        return abs(float(left) - float(right)) <= 1e-3
    return int(left) == int(right)


def execute(code, timeout):
    safe_builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum,
        "len": len, "range": range, "enumerate": enumerate,
        "int": int, "float": float, "str": str, "bool": bool,
        "round": round,
    }
    namespace = {"__builtins__": safe_builtins}
    with time_limit(timeout):
        with warnings.catch_warnings(), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore", SyntaxWarning)
            exec(code, namespace, namespace)
            return to_number(namespace["simple_math_problem"]())


def audit_record(record, candidates, execution, timeout):
    errors = []
    flags = []
    try:
        module = ast.parse(record["code"])
    except SyntaxError as exc:
        return [f"syntax:{exc.msg}"], flags

    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if len(module.body) != 1 or len(functions) != 1:
        errors.append("module_contract")
        return errors, flags
    function = functions[0]
    if (function.name != "simple_math_problem" or function.args.args
            or function.args.posonlyargs or function.args.kwonlyargs
            or function.args.vararg or function.args.kwarg
            or function.decorator_list):
        errors.append("function_contract")
    if ast.get_docstring(function, clean=False) != record["question"]:
        errors.append("docstring_question_mismatch")
    if any(isinstance(node, FORBIDDEN_NODES) for node in ast.walk(module)):
        errors.append("forbidden_ast")
    for call in (node for node in ast.walk(module) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Name) or call.func.id not in SAFE_CALLS:
            errors.append("unsupported_call")
            break
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    if len(returns) != 1:
        errors.append("return_count")
    assignments = [
        node for node in function.body
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
    ]
    if not assignments:
        flags.append("no_intermediate_assignment")
    if returns and isinstance(returns[0].value, ast.Constant):
        flags.append("literal_return")

    row_id = record["mathinstruct_row_id"]
    candidate = candidates.get(row_id)
    verified = execution.get(row_id)
    if candidate is None or verified is None:
        errors.append("missing_provenance")
    else:
        if verified.get("status") != "verified":
            errors.append("provenance_not_verified")
        expected = (
            candidate["gsm8k_train_id"], candidate["question"], candidate["code"]
        )
        actual = (
            record["gsm8k_train_id"], record["question"], record["code"]
        )
        if actual != expected:
            errors.append("provenance_content_mismatch")

    try:
        predicted = execute(record["code"], timeout)
        if not numbers_equal(predicted, to_number(record["answer"])):
            errors.append("final_execution_mismatch")
    except Exception as exc:  # Report the exact final artifact failure.
        errors.append(f"execution:{type(exc).__name__}")
    return sorted(set(errors)), sorted(set(flags))


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data/gsm8k_pot")
    parser.add_argument("--audit-dir", type=Path,
                        default=ROOT / "data/mathinstruct-audit")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/sft-review/gsm8k-pot-final-audit.json")
    parser.add_argument("--timeout", type=float, default=0.25)
    args = parser.parse_args()

    candidates = {
        row["row_id"]: row for row in load_jsonl(
            args.audit_dir / "verified-audit-candidates.jsonl")
    }
    execution = {
        row["row_id"]: row for row in load_jsonl(
            args.audit_dir / "execution-audit.jsonl")
    }
    split_rows = {
        split: load_jsonl(args.data_dir / f"sft_{split}.jsonl")
        for split in ("train", "validation")
    }
    errors = Counter()
    flags = Counter()
    flagged_examples = []
    accepted = 0
    ids = {}
    for split, rows in split_rows.items():
        ids[split] = {row["gsm8k_train_id"] for row in rows}
        if len(ids[split]) != len(rows):
            errors[f"{split}:duplicate_question_id"] += len(rows) - len(ids[split])
        for row in rows:
            row_errors, row_flags = audit_record(
                row, candidates, execution, args.timeout)
            if not row_errors:
                accepted += 1
            errors.update(f"{split}:{error}" for error in row_errors)
            flags.update(f"{split}:{flag}" for flag in row_flags)
            if row_flags and len(flagged_examples) < 20:
                flagged_examples.append({
                    "split": split,
                    "sample_id": row["sample_id"],
                    "flags": row_flags,
                    "code": row["code"],
                })
    overlap = ids["train"] & ids["validation"]
    if overlap:
        errors["split_overlap"] = len(overlap)

    result = {
        "status": "passed" if not errors else "failed",
        "policy": (
            "Audit and quarantine only; do not rewrite verified mathematical "
            "logic or remove docstrings."
        ),
        "rows": {split: len(rows) for split, rows in split_rows.items()},
        "accepted_rows": accepted,
        "errors": dict(sorted(errors.items())),
        "style_flags_not_removed": dict(sorted(flags.items())),
        "flagged_examples": flagged_examples,
        "files": {
            f"sft_{split}.jsonl": sha256(
                args.data_dir / f"sft_{split}.jsonl")
            for split in ("train", "validation")
        },
        "checks": [
            "exact MathInstruct audit provenance and verified status",
            "single no-argument simple_math_problem function",
            "docstring exactly matches the GSM8K question",
            "restricted AST and calls",
            "final program executes and matches the stored gold answer",
            "unique, disjoint train and validation question IDs",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
