#!/usr/bin/env python3
"""Summarize raw/EMA and fixed training-subset GSM8K diagnostics."""

import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# sandbox_gsm8k only needs these NumPy scalar type aliases for this analysis.
np_stub = types.ModuleType("numpy")
np_stub.integer = int
np_stub.floating = float
sys.modules.setdefault("numpy", np_stub)
from sandbox_gsm8k import _extract_code, evaluate_samples


def load_records(path):
    payload = json.loads(path.read_text())
    return payload, payload["records"]


def summarize(payload, records):
    parseable = 0
    unpaired_quotes = 0
    for row in records:
        response = row["response"]
        try:
            compile(_extract_code(response), "<sample>", "exec")
            parseable += 1
        except (SyntaxError, ValueError, TypeError):
            pass
        if response.count('"""') % 2 or response.count("'''") % 2:
            unpaired_quotes += 1
    return {
        "correct": payload["num_correct"],
        "total": payload["num_total"],
        "accuracy": payload["accuracy"],
        "parseable": parseable,
        "parseable_rate": parseable / len(records),
        "unpaired_triple_quotes": unpaired_quotes,
        "mean_response_chars": sum(len(r["response"]) for r in records) / len(records),
    }


def paired(left, right, left_correct, right_correct):
    assert [r["prompt"] for r in left] == [r["prompt"] for r in right]
    return {
        "left_correct_to_right_correct": sum(a and b for a, b in zip(left_correct, right_correct)),
        "left_correct_to_right_wrong": sum(a and not b for a, b in zip(left_correct, right_correct)),
        "left_wrong_to_right_correct": sum(not a and b for a, b in zip(left_correct, right_correct)),
        "both_wrong": sum(not a and not b for a, b in zip(left_correct, right_correct)),
        "same_response": sum(a["response"] == b["response"] for a, b in zip(left, right)),
    }


def main():
    parser_root = ROOT / "eval_runs/gsm8k-sft-diagnostics/20261022-lora-result-lr1e5-diagnostics"
    summary = {"validation": {}, "train_subset": {}}
    records = {"validation": {}, "train_subset": {}}
    correct = {"validation": {}, "train_subset": {}}

    paths = {
        "validation": {
            "base_ema": ROOT / "eval_runs/gsm8k-sft-validation/20261022-lora-result-lr1e5-validation/step-0-ema-steps32-seed1/results.json",
            "step200_ema": parser_root / "validation/step-200-ema-steps32-seed1/results.json",
            "step200_raw": parser_root / "validation/step-200-raw-steps32-seed1/results.json",
            "step711_ema": parser_root / "validation/step-711-ema-steps32-seed1/results.json",
            "step711_raw": parser_root / "validation/step-711-raw-steps32-seed1/results.json",
        },
        "train_subset": {
            "base_ema": parser_root / "train-subset/base-ema-steps32-seed1/results.json",
            "step200_ema": parser_root / "train-subset/step-200-ema-steps32-seed1/results.json",
            "step200_raw": parser_root / "train-subset/step-200-raw-steps32-seed1/results.json",
            "step711_ema": parser_root / "train-subset/step-711-ema-steps32-seed1/results.json",
            "step711_raw": parser_root / "train-subset/step-711-raw-steps32-seed1/results.json",
        },
    }

    for split, named_paths in paths.items():
        for name, path in named_paths.items():
            payload, rows = load_records(path)
            records[split][name] = rows
            correct[split][name] = [
                evaluate_samples(r["response"], r["response_ground_truth"], 1.0)
                for r in rows
            ]
            measured = sum(correct[split][name])
            assert measured == payload["num_correct"], (split, name, measured, payload["num_correct"])
            summary[split][name] = summarize(payload, rows)

    paired_rows = {"validation": {}, "train_subset": {}}
    for split in paths:
        models = list(paths[split])
        for i, left_name in enumerate(models):
            for right_name in models[i + 1:]:
                paired_rows[split][f"{left_name}_vs_{right_name}"] = paired(
                    records[split][left_name], records[split][right_name],
                    correct[split][left_name], correct[split][right_name])
    for name in ("step200_ema", "step200_raw", "step711_ema", "step711_raw"):
        paired_rows["validation"][f"base_ema_vs_{name}"] = paired(
            records["validation"]["base_ema"], records["validation"][name],
            correct["validation"]["base_ema"], correct["validation"][name])
    for name in ("step200_ema", "step200_raw", "step711_ema", "step711_raw"):
        paired_rows["train_subset"][f"base_ema_vs_{name}"] = paired(
            records["train_subset"]["base_ema"], records["train_subset"][name],
            correct["train_subset"]["base_ema"], correct["train_subset"][name])

    (parser_root / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (parser_root / "paired_analysis.json").write_text(json.dumps(paired_rows, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(json.dumps(paired_rows, indent=2))


if __name__ == "__main__":
    main()
