#!/usr/bin/env python3
"""Compare generated GSM8K PoT programs with their reference target structure."""
from __future__ import annotations

import ast
import csv
import json
import os
import re
import statistics
import sys
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = Path(os.environ.get(
    "EVAL_ROOT", ROOT / "eval_runs/gsm8k-sft-target-alignment"))
SETS = {
    "train-subset": (
        ROOT / "data/gsm8k_pot_result/diagnostics/train_eval_subset_seed20261022_n256.jsonl",
        EVAL_ROOT / "train-subset",
    ),
    "validation": (
        ROOT / "data/gsm8k_pot_result/sft_validation.jsonl",
        EVAL_ROOT / "validation",
    ),
}
MODELS = {
    "base_ema": "base-ema-steps32-seed1",
    "step100_ema": "step-100-ema-steps32-seed1",
    "step200_ema": "step-200-ema-steps32-seed1",
    "step711_ema": "step-711-ema-steps32-seed1",
}
OUT_CSV = Path(os.environ.get("ALIGNMENT_CSV", ROOT / "reports/gsm8k_pot_target_alignment_20261022.csv"))
OUT_MD = Path(os.environ.get("ALIGNMENT_MD", ROOT / "reports/gsm8k_pot_target_alignment_20261022.md"))

ARITH = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)


def extract_code(text: str) -> str:
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.I | re.S)
    if fence:
        text = fence.group(1)
    start = text.find("def ")
    return text[start:].strip() if start >= 0 else text.strip()


def evaluator_recovery(code: str) -> str:
    """Mirror sandbox_gsm8k's valid-prefix recovery for a late syntax error."""
    try:
        compile(code, "<sample>", "exec")
        return code
    except SyntaxError as exc:
        if exc.lineno is None or exc.lineno <= 1:
            return code
        prefix = "\n".join(code.splitlines()[:exc.lineno - 1]).strip()
        try:
            compile(prefix, "<sample-prefix>", "exec")
            return prefix
        except SyntaxError:
            return code


def inspect(code: str) -> dict:
    result = {
        "parseable": False, "function_count": 0, "single_expected_function": False,
        "return_count": 0, "returns_result": False, "assigns_result": False,
        "imports": 0, "control_flow": 0, "assignment_count": 0,
        "dependency_depth": 0, "arithmetic_ops": {},
    }
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return result
    result["parseable"] = True
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    result["function_count"] = len(funcs)
    result["single_expected_function"] = (
        len(tree.body) == 1 and len(funcs) == 1
        and isinstance(funcs[0], ast.FunctionDef)
        and funcs[0].name == "simple_math_problem"
    )
    result["imports"] = sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
    result["control_flow"] = sum(isinstance(n, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.Try)) for n in ast.walk(tree))
    result["assignment_count"] = sum(isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)) for n in ast.walk(tree))
    result["arithmetic_ops"] = dict(Counter(type(n.op).__name__ for n in ast.walk(tree) if isinstance(n, ast.BinOp)))
    if not funcs:
        return result
    fn = funcs[0]
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    result["return_count"] = len(returns)
    result["returns_result"] = len(returns) == 1 and isinstance(returns[0].value, ast.Name) and returns[0].value.id == "result"
    assigned = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    result["assigns_result"] = "result" in assigned

    depths: dict[str, int] = {}
    max_depth = 0
    def names(expr):
        return {n.id for n in ast.walk(expr) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    def visit(stmts):
        nonlocal max_depth
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                used = names(stmt.value)
                d = max((depths.get(n, 0) for n in used), default=0) + (1 if used else 0)
                for target in stmt.targets:
                    for n in ast.walk(target):
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                            depths[n.id] = d
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                used = names(stmt.value)
                d = max((depths.get(n, 0) for n in used), default=0) + (1 if used else 0)
                for n in ast.walk(stmt.target):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        depths[n.id] = d
            elif isinstance(stmt, ast.AugAssign):
                for n in ast.walk(stmt.target):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        depths[n.id] = depths.get(n.id, 0) + 1
            elif isinstance(stmt, ast.Return) and stmt.value:
                max_depth = max(max_depth, *(depths.get(n, 0) for n in names(stmt.value)), 0)
            elif isinstance(stmt, ast.If):
                visit(stmt.body); visit(stmt.orelse)
            elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.Try)):
                for key in ("body", "orelse", "finalbody"):
                    visit(getattr(stmt, key, []))
                for handler in getattr(stmt, "handlers", []):
                    visit(handler.body)
    visit(fn.body)
    result["dependency_depth"] = max(max_depth, max(depths.values(), default=0))
    return result


def op_jaccard(a, b):
    ca, cb = Counter(a), Counter(b)
    union = sum((ca | cb).values())
    return sum((ca & cb).values()) / union if union else 1.0


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_scorer():
    # sandbox_gsm8k only needs these NumPy scalar aliases for its numeric checks.
    if "numpy" not in sys.modules:
        np = types.ModuleType("numpy")
        np.integer = int
        np.floating = float
        sys.modules["numpy"] = np
    sys.path.insert(0, str(ROOT))
    import sandbox_gsm8k
    return sandbox_gsm8k.evaluate_samples


def main():
    score = load_scorer()
    all_rows = []
    for split, (data_path, eval_dir) in SETS.items():
        targets = load_jsonl(data_path)
        for model, dirname in MODELS.items():
            result_path = EVAL_ROOT / split / dirname / "results.json"
            records = json.loads(result_path.read_text())["records"]
            if len(records) != len(targets):
                raise ValueError(f"{split}/{model}: target/eval count mismatch")
            for i, rec in enumerate(records):
                target = targets[i]
                if rec["prompt"] != target["question"]:
                    raise ValueError(f"Question order mismatch in {split} at {i}")
                tgt = inspect(target["code"])
                strict_code = extract_code(rec["response"])
                strict_gen = inspect(strict_code)
                generated_code = evaluator_recovery(strict_code)
                gen = inspect(generated_code)
                row = {
                    "split": split, "model": model, "sample_id": target["sample_id"],
                    "gsm8k_train_id": target.get("gsm8k_train_id"),
                    "correct": score(rec["response"], rec["response_ground_truth"], timeout_s=2),
                    "target_parseable": tgt["parseable"], "strict_generated_parseable": strict_gen["parseable"],
                    "evaluator_recovered_parseable": gen["parseable"],
                    "single_expected_function": gen["single_expected_function"],
                    "returns_result": gen["returns_result"], "assigns_result": gen["assigns_result"],
                    "contract_pass": gen["single_expected_function"] and gen["returns_result"] and gen["assigns_result"],
                    "imports": gen["imports"], "generated_control_flow": gen["control_flow"],
                    "target_control_flow": tgt["control_flow"],
                    "target_assignment_count": tgt["assignment_count"],
                    "generated_assignment_count": gen["assignment_count"],
                    "target_dependency_depth": tgt["dependency_depth"],
                    "generated_dependency_depth": gen["dependency_depth"],
                    "operator_multiset_jaccard": op_jaccard(tgt["arithmetic_ops"].items(), gen["arithmetic_ops"].items()),
                    "target_ops": json.dumps(tgt["arithmetic_ops"], sort_keys=True),
                    "generated_ops": json.dumps(gen["arithmetic_ops"], sort_keys=True),
                    "question": rec["prompt"], "reference_code": target["code"],
                    "generated_code": strict_code, "evaluator_recovered_code": generated_code,
                }
                all_rows.append(row)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader(); w.writerows(all_rows)

    lines = ["# GSM8K PoT target-structure alignment", "",
             "Generated programs are compared with the reference code for each question. This measures code-form imitation, not mathematical equivalence: correct solutions may legitimately use different operators or intermediate variables. Generation is one sampled response per question (32 denoising steps, seed 1).", "",
             "`strict generated parseable` means the full generated function parses as-is. `evaluator recovered parseable` mirrors the project's evaluator, which can cut trailing malformed output after an earlier valid prefix. `contract_pass` is measured on that evaluator-recovered code and means one top-level `simple_math_problem` function, an assignment to `result`, and exactly one `return result`. `operator_multiset_jaccard` compares arithmetic operator counts; 1 means the multisets match. Neither metric alone proves mathematical equivalence.", "",
             "| Split | Model | N | Correct | Strict parseable | Evaluator recovered parseable | Contract pass | Mean op Jaccard | Mean target/generated assignments | Mean target/generated dependency depth |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in SETS:
        for model in MODELS:
            subset = [r for r in all_rows if r["split"] == split and r["model"] == model]
            n = len(subset)
            def rate(key): return sum(bool(r[key]) for r in subset) / n if n else 0
            oj = statistics.mean(r["operator_multiset_jaccard"] for r in subset)
            ta = statistics.mean(r["target_assignment_count"] for r in subset)
            ga = statistics.mean(r["generated_assignment_count"] for r in subset)
            td = statistics.mean(r["target_dependency_depth"] for r in subset)
            gd = statistics.mean(r["generated_dependency_depth"] for r in subset)
            correct = sum(r["correct"] for r in subset)
            lines.append(f"| {split} | {model} | {n} | {correct} ({correct/n:.2%}) | {rate('strict_generated_parseable'):.2%} | {rate('evaluator_recovered_parseable'):.2%} | {rate('contract_pass'):.2%} | {oj:.3f} | {ta:.2f} / {ga:.2f} | {td:.2f} / {gd:.2f} |")
    lines += ["", "## Structural metrics split by answer correctness", "",
              "| Split | Model | Answer group | N | Contract pass | Mean op Jaccard |", "|---|---|---|---:|---:|---:|"]
    for split in SETS:
        for model in MODELS:
            group = [r for r in all_rows if r["split"] == split and r["model"] == model]
            for correct, label in ((True, "correct"), (False, "wrong")):
                sub = [r for r in group if r["correct"] == correct]
                n = len(sub)
                cp = sum(r["contract_pass"] for r in sub) / n if n else 0
                oj = statistics.mean(r["operator_multiset_jaccard"] for r in sub) if n else 0
                lines.append(f"| {split} | {model} | {label} | {n} | {cp:.2%} | {oj:.3f} |")
    lines += ["", "On the validation split, every answer counted correct also passed this `result`/`return result` contract, but most contract-conforming programs were still numerically wrong. Correct answers have much higher operator-multiset overlap than wrong answers, as expected when the generated calculation plan matches the reference. At step 711 both the contract rate and overall operator overlap are slightly lower than Base on validation; the train subset also shows no structural or accuracy gain. This is preliminary evidence that 711-step SFT has not improved target imitation, while format conformity alone is not enough to fix arithmetic correctness.", "", "Use the companion CSV to inspect individual failures. A low contract rate points to format imitation; a high contract rate with low accuracy and weak operator/dependency alignment points to calculation planning or operand/value errors; high structural similarity with wrong answers points further toward grounding/arithmetic errors. Inspect examples manually before assigning a cause.", ""]
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_CSV}\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
