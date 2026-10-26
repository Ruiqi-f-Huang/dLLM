#!/usr/bin/env python3
"""Analyze GSM8K PoT validation code complexity vs Base/step-711 correctness.

Difficulty is an exploratory code-complexity proxy derived only from reference
programs. It is not a gold measure of mathematical reasoning difficulty.
"""
from __future__ import annotations

import ast
import csv
import json
import math
import statistics
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "data/gsm8k_pot_result/sft_validation.jsonl"
EVAL_ROOT = ROOT / "eval_runs/gsm8k-sft-validation/20261022-lora-result-lr1e5-validation"
BASE_RESULTS = EVAL_ROOT / "step-0-ema-steps32-seed1/results.json"
SFT_RESULTS = EVAL_ROOT / "step-711-ema-steps32-seed1/results.json"
OUT_CSV = ROOT / "reports/gsm8k_pot_validation_difficulty_20261022.csv"
OUT_MD = ROOT / "reports/gsm8k_pot_validation_difficulty_20261022.md"
OUT_JSON = ROOT / "reports/gsm8k_pot_validation_difficulty_20261022.json"


def load_sandbox():
    # The project scorer imports NumPy, which is absent in the local analysis
    # environment. Its use here is limited to integer/floating type checks.
    if "numpy" not in sys.modules:
        np = types.ModuleType("numpy")
        np.integer = int
        np.floating = float
        sys.modules["numpy"] = np
    sys.path.insert(0, str(ROOT))
    import sandbox_gsm8k
    return sandbox_gsm8k


ARITH = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
ALL_OPS = ARITH + (ast.UAdd, ast.USub, ast.Not, ast.Eq, ast.NotEq, ast.Lt,
                    ast.LtE, ast.Gt, ast.GtE, ast.And, ast.Or)
CONTROL = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.Try,
           ast.ExceptHandler, ast.Match)


def code_features(code: str) -> dict:
    out = {
        "ast_parseable": False, "ast_nodes": 0, "arithmetic_ops": 0,
        "all_ops": 0, "control_flow": 0, "dependency_depth": 0,
        "numeric_types": 0, "float_literals": 0, "advanced_ops": 0,
        "function_count": 0, "call_count": 0, "assigned_names": 0,
        "dead_assignments": 0,
    }
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return out
    out["ast_parseable"] = True
    nodes = list(ast.walk(tree))
    out["ast_nodes"] = len(nodes)
    out["arithmetic_ops"] = sum(isinstance(n, ARITH) for n in nodes)
    out["all_ops"] = sum(isinstance(n, ALL_OPS) for n in nodes)
    out["control_flow"] = sum(isinstance(n, CONTROL) for n in nodes)
    out["function_count"] = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in nodes)
    out["call_count"] = sum(isinstance(n, ast.Call) for n in nodes)
    nums = [n.value for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)]
    num_types = {"int" if isinstance(n, int) else "float" for n in nums}
    out["numeric_types"] = len(num_types)
    out["float_literals"] = sum(isinstance(n, float) for n in nums)
    out["advanced_ops"] = sum(isinstance(n, (ast.FloorDiv, ast.Mod, ast.Pow)) for n in nodes)

    funcs = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == "simple_math_problem"]
    fn = funcs[0] if funcs else next((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None:
        return out

    # Track how many variable-to-variable assignment layers lead into each
    # named value. Constants start at depth 0; each dependent assignment adds 1.
    depths: dict[str, int] = {}
    assigned_order: list[tuple[str, int]] = []
    max_return_depth = 0

    def expr_names(expr: ast.AST) -> set[str]:
        return {n.id for n in ast.walk(expr) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

    def visit_statements(stmts):
        nonlocal max_return_depth
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                names = [t.id for target in stmt.targets for t in ast.walk(target) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)]
                used = expr_names(stmt.value)
                dep = max((depths.get(n, 0) for n in used), default=0)
                value_depth = dep + (1 if used else 0)
                for name in names:
                    depths[name] = value_depth
                    assigned_order.append((name, len(assigned_order)))
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                names = [n.id for n in ast.walk(stmt.target) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
                used = expr_names(stmt.value)
                value_depth = max((depths.get(n, 0) for n in used), default=0) + (1 if used else 0)
                for name in names:
                    depths[name] = value_depth
                    assigned_order.append((name, len(assigned_order)))
            elif isinstance(stmt, ast.AugAssign):
                names = [n.id for n in ast.walk(stmt.target) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
                used = expr_names(stmt.value) | {n.id for n in ast.walk(stmt.target) if isinstance(n, ast.Name)}
                value_depth = max((depths.get(n, 0) for n in used), default=0) + 1
                for name in names:
                    depths[name] = value_depth
                    assigned_order.append((name, len(assigned_order)))
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                used = expr_names(stmt.value)
                max_return_depth = max(max_return_depth, max((depths.get(n, 0) for n in used), default=0))
            elif isinstance(stmt, ast.If):
                visit_statements(stmt.body)
                visit_statements(stmt.orelse)
            elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.Try)):
                for field in ("body", "orelse", "finalbody"):
                    visit_statements(getattr(stmt, field, []))
                for handler in getattr(stmt, "handlers", []):
                    visit_statements(handler.body)
    visit_statements(fn.body)
    out["dependency_depth"] = max(max_return_depth, max(depths.values(), default=0))
    out["assigned_names"] = len(assigned_order)
    loaded = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    out["dead_assignments"] = sum(name not in loaded for name, _ in assigned_order)
    return out


def average_percentile(values):
    # Midrank percentile, stable in the presence of ties.
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        pct = ((i + j - 1) / 2) / max(1, len(order) - 1)
        for k in range(i, j):
            ranks[order[k]] = pct
        i = j
    return ranks


def pct(values, p):
    x = sorted(values)
    pos = (len(x) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return x[lo] + (x[hi] - x[lo]) * (pos - lo)


def main():
    scorer = load_sandbox()
    rows = [json.loads(line) for line in VALIDATION.read_text().splitlines() if line.strip()]
    base = json.loads(BASE_RESULTS.read_text())["records"]
    sft = json.loads(SFT_RESULTS.read_text())["records"]
    if not (len(rows) == len(base) == len(sft)):
        raise ValueError(f"Row count mismatch: data={len(rows)}, base={len(base)}, sft={len(sft)}")
    data = []
    for i, (row, b, s) in enumerate(zip(rows, base, sft)):
        if row["question"] != b["prompt"] or row["question"] != s["prompt"]:
            raise ValueError(f"Question ordering mismatch at {i}")
        feat = code_features(row["code"])
        data.append({**{k: row.get(k) for k in ("sample_id", "gsm8k_train_id", "mathinstruct_row_id")},
                     "question": row["question"], **feat,
                     "base_correct": scorer.evaluate_samples(b["response"], row["reference_answer"], timeout_s=2),
                     "step711_correct": scorer.evaluate_samples(s["response"], row["reference_answer"], timeout_s=2)})

    feature_keys = ["arithmetic_ops", "dependency_depth", "control_flow", "advanced_ops", "numeric_types"]
    ranks = [average_percentile([r[k] for r in data]) for k in feature_keys]
    for i, row in enumerate(data):
        row["complexity_score"] = sum(rank[i] for rank in ranks) / len(ranks)
    scores = [r["complexity_score"] for r in data]
    low, high = pct(scores, 1/3), pct(scores, 2/3)
    for r in data:
        r["difficulty"] = "easy" if r["complexity_score"] <= low else ("medium" if r["complexity_score"] <= high else "hard")
        r["transition"] = ("both_correct" if r["base_correct"] and r["step711_correct"] else
                           "base_only" if r["base_correct"] else
                           "step711_only" if r["step711_correct"] else "both_wrong")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)

    bins = ["easy", "medium", "hard"]
    summary = {"n": len(data), "thresholds": {"easy_max_score": low, "medium_max_score": high},
               "features": {}, "bins": {}, "transitions": {},
               "overall": {"base": sum(r["base_correct"] for r in data),
                           "step711": sum(r["step711_correct"] for r in data)}}
    for key in feature_keys + ["ast_nodes", "all_ops", "float_literals", "dead_assignments"]:
        vals = [r[key] for r in data]
        summary["features"][key] = {"mean": statistics.mean(vals), "median": statistics.median(vals),
                                     "p90": pct(vals, .9), "max": max(vals)}
    for bin_name in bins:
        subset = [r for r in data if r["difficulty"] == bin_name]
        b = sum(r["base_correct"] for r in subset)
        s = sum(r["step711_correct"] for r in subset)
        summary["bins"][bin_name] = {"n": len(subset), "base_correct": b,
            "base_accuracy": b / len(subset) if subset else None,
            "step711_correct": s, "step711_accuracy": s / len(subset) if subset else None,
            "delta_pp": (s - b) / len(subset) * 100 if subset else None}
        summary["transitions"][bin_name] = dict(Counter(r["transition"] for r in subset))
    summary["overall"]["base_accuracy"] = summary["overall"]["base"] / len(data)
    summary["overall"]["step711_accuracy"] = summary["overall"]["step711"] / len(data)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    lines = ["# GSM8K PoT validation: code complexity vs Base/step-711", "",
        f"Validation set: {len(data)} questions. Base and step-711 are scored using the project's GSM8K sandbox (EMA, 32 denoising steps, seed 1).",
        "",
        "Difficulty is an exploratory composite: average midrank percentile across arithmetic operator count, variable dependency depth, control-flow count, advanced arithmetic operators (//, %, **), and numeric literal type diversity. Ties are kept together at the 33rd/67th percentile cutoffs, so bin sizes need not be equal. The score measures reference-code complexity, not validated mathematical difficulty.", "",
        f"Cutoffs: easy ≤ {low:.4f}; medium ≤ {high:.4f}; hard > {high:.4f}.", "",
        "| Difficulty | N | Base correct | Base acc. | 711 correct | 711 acc. | 711 − Base |", "|---|---:|---:|---:|---:|---:|---:|"]
    for bn in bins:
        x = summary["bins"][bn]
        lines.append(f"| {bn} | {x['n']} | {x['base_correct']} | {100*x['base_accuracy']:.2f}% | {x['step711_correct']} | {100*x['step711_accuracy']:.2f}% | {x['delta_pp']:+.2f} pp |")
    lines += ["", f"Overall: Base {summary['overall']['base']}/{len(data)} ({100*summary['overall']['base_accuracy']:.2f}%); step-711 {summary['overall']['step711']}/{len(data)} ({100*summary['overall']['step711_accuracy']:.2f}%).", "",
        "| Difficulty | Both correct | Base only | 711 only | Both wrong |", "|---|---:|---:|---:|---:|"]
    for bn in bins:
        t = summary["transitions"][bn]
        lines.append(f"| {bn} | {t.get('both_correct',0)} | {t.get('base_only',0)} | {t.get('step711_only',0)} | {t.get('both_wrong',0)} |")
    lines += ["", "## Complexity feature summary", "", "| Feature | Mean | Median | P90 | Max |", "|---|---:|---:|---:|---:|"]
    for key, vals in summary["features"].items():
        lines.append(f"| {key} | {vals['mean']:.2f} | {vals['median']:.2f} | {vals['p90']:.2f} | {vals['max']} |")
    lines += ["", "## What the split says", "",
        "Both models' correct answers are concentrated in the easy bin (Base: 33/54; step-711: 28/46), and accuracy falls as reference-code complexity rises. However, step-711 does not show a selective collapse on hard questions: its change is −2.31 pp on easy, −1.46 pp on medium, and 0.00 pp on hard. The hard-bin equality is based on only 9 correct answers per model and is strongly affected by the low-accuracy floor. Paired transitions show 11 Base-only versus 3 step-711-only overall, so the observed degradation is small and spread across bins rather than clearly concentrated in one difficulty band.", "",
        "The current features are weakly discriminative for this data: 630/632 reference programs have no control flow, advanced operators are rare, and most programs are short straight-line arithmetic. Thus this is useful as a diagnostic, but not strong evidence that curriculum learning is the main SFT bottleneck. In addition, the train-subset diagnostic found no measurable fit gain at step 711 (Base 23/256; step-711 18/256), which makes target/optimization alignment a higher-priority issue than ordering examples by difficulty.", "",
        "## Recommended next experiment", "",
        "1. Keep the current result-return data transform and existing LoRA configuration fixed. Compute these same features for the 5,682 training examples, but define bins and thresholds from the training split only. Exclude invalid programs and inspect a small sample from each bin to ensure that high AST complexity reflects useful reasoning rather than verbose or dead code.",
        "2. Compare two runs with identical initialization, seed, optimizer, effective batch size, total update count, and evaluation settings: (A) the current uniform shuffle; (B) curriculum with a short easy-to-hard warm-up followed by mixed sampling. A practical B schedule is 20% easy / 60% medium / 20% hard during the first 20% of updates, then 1/3 from each bin for the remaining updates. Keep total example exposure controlled and log per-bin train loss.",
        "3. Evaluate Base and both runs on this same validation split at the predeclared 100/200/711 milestones. Report overall and per-bin accuracy, paired Base→SFT transitions, parseability, and target-structure match. Do not use validation-bin results to resample training examples or tune the schedule; keep GSM8K Test untouched for the final comparison.",
        "4. Treat curriculum as supported only if it beats uniform shuffle on overall validation accuracy across more than one seed or gives a consistent gain in the predeclared hard bin without materially harming easy/medium. Given the current very low accuracy and no train-subset fit gain, first inspect generated-vs-target code alignment and short-batch training loss; if those reveal weak imitation, fix the objective/data path before spending a full run on curriculum.",
        "", "Question-level features, difficulty labels, and per-model correctness are in the companion CSV. This analysis does not use validation questions for training or schedule selection.", ""]
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_CSV}\nWrote {OUT_MD}\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
