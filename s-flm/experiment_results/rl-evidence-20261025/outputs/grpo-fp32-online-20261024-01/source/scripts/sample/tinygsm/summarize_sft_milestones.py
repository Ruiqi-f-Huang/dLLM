"""Summarize accuracy and Python-format diagnostics for milestone results."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from sandbox_gsm8k import _extract_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    summaries = []
    for path in sorted(args.output_root.glob("step-*/results.json")):
        data = json.loads(path.read_text())
        responses = [record["response"] for record in data["records"]]
        parseable = 0
        for response in responses:
            try:
                compile(_extract_code(response), "<sample>", "exec")
                parseable += 1
            except (SyntaxError, ValueError, TypeError):
                pass
        unpaired_quotes = sum(
            response.count('"""') % 2 == 1 or response.count("'''") % 2 == 1
            for response in responses)
        step = int(path.parent.name.split("-", 2)[1])
        summaries.append({
            "step": step,
            "num_correct": data["num_correct"],
            "num_total": data["num_total"],
            "accuracy": data["accuracy"],
            "python_parseable": parseable,
            "python_parseable_rate": parseable / len(responses),
            "unpaired_triple_quotes": unpaired_quotes,
            "unpaired_triple_quotes_rate": unpaired_quotes / len(responses),
            "results_file": str(path),
        })
    summaries.sort(key=lambda row: row["step"])
    output = args.output_root / "milestone_summary.json"
    output.write_text(json.dumps(summaries, indent=2))
    print("step  correct/total  accuracy  parseable  unpaired_quotes")
    for row in summaries:
        print(f"{row['step']:>4}  {row['num_correct']:>4}/{row['num_total']:<4}  "
              f"{row['accuracy'] * 100:>7.2f}%  "
              f"{row['python_parseable']:>4}/{row['num_total']:<4}  "
              f"{row['unpaired_triple_quotes']:>4}/{row['num_total']:<4}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

