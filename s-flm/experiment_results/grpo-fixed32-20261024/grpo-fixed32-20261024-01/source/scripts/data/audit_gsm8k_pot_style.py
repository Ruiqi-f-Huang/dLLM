"""Measure GSM8K PoT format and complexity without rewriting the dataset."""

import argparse
import ast
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SAFE_NAMES = {
    "abs", "min", "max", "sum", "len", "range", "enumerate",
    "int", "float", "str", "bool", "round",
}
CONTROL_NODES = (
    ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
)


def load_rows(data_dir):
    rows = []
    for split in ("train", "validation"):
        path = data_dir / f"sft_{split}.jsonl"
        for line in path.read_text().splitlines():
            if line:
                row = json.loads(line)
                row["_split"] = split
                rows.append(row)
    return rows


def percentile(values, probability):
    values = sorted(values)
    return values[min(len(values) - 1,
                      math.ceil(probability * len(values)) - 1)]


def distribution(values):
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def target_names(target):
    return {
        node.id for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def analyze(row):
    tree = ast.parse(row["code"])
    top_functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    function = top_functions[0]
    body = list(function.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    body_module = ast.Module(body=body, type_ignores=[])
    nodes = list(ast.walk(function))

    assigned = set()
    for node in ast.walk(body_module):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            assigned.update(target_names(target))
    loaded = {
        node.id for node in ast.walk(body_module)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    defined = {
        node.id for node in ast.walk(body_module)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    undefined = sorted(loaded - defined - SAFE_NAMES)
    unused = sorted(assigned - loaded)

    final_named_return = False
    strict_result_return = False
    if (len(body) >= 2
            and isinstance(body[-2], (ast.Assign, ast.AnnAssign))
            and isinstance(body[-1], ast.Return)
            and isinstance(body[-1].value, ast.Name)):
        target = None
        if (isinstance(body[-2], ast.Assign)
                and len(body[-2].targets) == 1
                and isinstance(body[-2].targets[0], ast.Name)):
            target = body[-2].targets[0].id
        elif (isinstance(body[-2], ast.AnnAssign)
              and isinstance(body[-2].target, ast.Name)):
            target = body[-2].target.id
        final_named_return = target == body[-1].value.id
        strict_result_return = final_named_return and target == "result"
    direct_return = bool(
        body and isinstance(body[-1], ast.Return)
        and not isinstance(body[-1].value, ast.Name))

    metrics = {
        "total_tokens": row["unpadded_total_tokens"],
        "code_chars": len(row["code"]),
        "nonblank_lines": sum(bool(line.strip())
                              for line in row["code"].splitlines()),
        "body_statements": len(body),
        "ast_nodes": len(nodes),
        "assignments": sum(isinstance(node, (
            ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr))
                           for node in nodes),
        "operators": sum(isinstance(node, (
            ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare))
                         for node in nodes),
    }
    tail_flags = []
    if metrics["total_tokens"] > 384:
        tail_flags.append("total_tokens>384")
    if metrics["ast_nodes"] > 85:
        tail_flags.append("ast_nodes>85")
    if metrics["body_statements"] > 11:
        tail_flags.append("body_statements>11")
    if metrics["operators"] > 9:
        tail_flags.append("operators>9")

    return {
        "sample_id": row["sample_id"],
        "split": row["_split"],
        "top_level_contract": len(tree.body) == 1 and len(top_functions) == 1,
        "imports": any(isinstance(node, (ast.Import, ast.ImportFrom))
                       for node in ast.walk(tree)),
        "prints": any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "print" for node in nodes),
        "return_count": sum(isinstance(node, ast.Return) for node in nodes),
        "undefined_names": undefined,
        "unused_assignments": unused,
        "control_flow": any(isinstance(node, CONTROL_NODES)
                            for node in nodes),
        "strict_result_return": strict_result_return,
        "final_named_return": final_named_return,
        "direct_return": direct_return,
        "tail_flags": tail_flags,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data/gsm8k_pot")
    parser.add_argument("--output", type=Path, default=(
        ROOT / "reports/sft-review/gsm8k-pot-style-audit.json"))
    args = parser.parse_args()

    records = [analyze(row) for row in load_rows(args.data_dir)]
    counts = Counter()
    split_counts = defaultdict(Counter)
    flag_sets = defaultdict(set)
    metric_values = defaultdict(list)
    review_records = []
    for record in records:
        split = record["split"]
        flags = []
        facts = {
            "top_level_contract_failure": not record["top_level_contract"],
            "has_import": record["imports"],
            "has_print": record["prints"],
            "return_count_not_one": record["return_count"] != 1,
            "undefined_name": bool(record["undefined_names"]),
            "unused_assignment": bool(record["unused_assignments"]),
            "control_flow": record["control_flow"],
            "strict_result_return": record["strict_result_return"],
            "equivalent_final_named_return": record["final_named_return"],
            "direct_return_expression": record["direct_return"],
            "tail_complexity": bool(record["tail_flags"]),
        }
        for name, present in facts.items():
            if present:
                counts[name] += 1
                split_counts[name][split] += 1
                flag_sets[name].add(record["sample_id"])
        for flag in record["tail_flags"]:
            counts[flag] += 1
            split_counts[flag][split] += 1
            flag_sets[flag].add(record["sample_id"])
        for name, value in record["metrics"].items():
            metric_values[name].append(value)
        if (facts["unused_assignment"] or facts["control_flow"]
                or facts["tail_complexity"]):
            flags.extend(record["tail_flags"])
            if facts["unused_assignment"]:
                flags.append("unused_assignment")
            if facts["control_flow"]:
                flags.append("control_flow")
            review_records.append({
                "sample_id": record["sample_id"],
                "split": split,
                "flags": sorted(flags),
                "unused_names": record["unused_assignments"],
                "metrics": record["metrics"],
            })

    def union_count(*names):
        return len(set().union(*(flag_sets[name] for name in names)))

    result = {
        "scope": {
            "train": sum(r["split"] == "train" for r in records),
            "validation": sum(r["split"] == "validation" for r in records),
            "total": len(records),
        },
        "definitions": {
            "strict_result_return": (
                "The final two non-docstring statements are exactly an "
                "assignment to result and return result."),
            "unused_assignment": (
                "A name assigned by Assign/AnnAssign/AugAssign is never loaded; "
                "loop iteration variables are excluded."),
            "control_flow": (
                "Contains if/for/while, a conditional expression, or a "
                "comprehension. This is a review flag, not an error."),
            "tail_complexity": (
                "Any of: total input+target tokens >384, function AST nodes "
                ">85, non-docstring body statements >11, or arithmetic/logic "
                "operators >9. These are approximately upper-tail cutoffs, "
                "not correctness failures."),
        },
        "counts": dict(sorted(counts.items())),
        "counts_by_split": {
            name: dict(sorted(value.items()))
            for name, value in sorted(split_counts.items())
        },
        "derived": {
            "strict_format_noncompliant": (
                len(records) - counts["strict_result_return"]),
            "unused_or_tail_review": union_count(
                "unused_assignment", "tail_complexity"),
            "control_or_tail_review": union_count(
                "control_flow", "tail_complexity"),
            "unused_or_control_or_tail_review": union_count(
                "unused_assignment", "control_flow", "tail_complexity"),
        },
        "distributions": {
            name: distribution(values)
            for name, values in sorted(metric_values.items())
        },
        "threshold_sensitivity": {
            "total_tokens": {
                f">{threshold}": sum(value > threshold for value in
                                      metric_values["total_tokens"])
                for threshold in (256, 320, 384, 448)
            },
            "ast_nodes": {
                f">{threshold}": sum(value > threshold for value in
                                      metric_values["ast_nodes"])
                for threshold in (75, 85, 100)
            },
            "body_statements": {
                f">{threshold}": sum(value > threshold for value in
                                      metric_values["body_statements"])
                for threshold in (10, 11, 12)
            },
            "operators": {
                f">{threshold}": sum(value > threshold for value in
                                      metric_values["operators"])
                for threshold in (8, 9, 12)
            },
        },
        "review_records": review_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "scope", "definitions", "counts", "counts_by_split", "derived",
        "distributions", "threshold_sensitivity")}, indent=2))


if __name__ == "__main__":
    main()
