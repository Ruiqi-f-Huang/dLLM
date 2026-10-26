"""Recompute transformed GSM8K PoT lengths with the checkpoint tokenizer."""

import argparse
import json
import math
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def percentile(values, probability):
    values = sorted(values)
    return values[min(len(values) - 1,
                      math.ceil(probability * len(values)) - 1)]


def stats(values):
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--separator", default=r"\n")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True)
    separator = tokenizer(
        args.separator, add_special_tokens=False).input_ids
    report = {
        "tokenizer": tokenizer.name_or_path,
        "separator": args.separator,
        "max_length": args.max_length,
        "splits": {},
    }
    all_lengths = []
    all_deltas = []
    too_long = []
    for split in ("train", "validation"):
        path = args.data_dir / f"sft_{split}.jsonl"
        lengths, deltas = [], []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line:
                continue
            row = json.loads(line)
            question = tokenizer(
                row["question"].strip(), add_special_tokens=False).input_ids
            code = tokenizer(
                row["code"].strip(), add_special_tokens=False).input_ids
            length = 1 + len(question) + len(separator) + len(code) + 1
            lengths.append(length)
            original = row.get("original_unpadded_total_tokens")
            if original is not None:
                deltas.append(length - original)
            if length > args.max_length:
                too_long.append({
                    "split": split,
                    "line": line_number,
                    "sample_id": row["sample_id"],
                    "tokens": length,
                })
        report["splits"][split] = {
            "rows": len(lengths),
            "lengths": stats(lengths),
            "delta_from_source": stats(deltas) if deltas else None,
        }
        all_lengths.extend(lengths)
        all_deltas.extend(deltas)
    report["all"] = {
        "rows": len(all_lengths),
        "lengths": stats(all_lengths),
        "delta_from_source": stats(all_deltas) if all_deltas else None,
        "too_long": too_long,
    }
    output = args.output or args.data_dir / "tokenization_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if too_long:
        raise SystemExit(f"{len(too_long)} transformed samples exceed max length")


if __name__ == "__main__":
    main()
