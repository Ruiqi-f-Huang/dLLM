#!/usr/bin/env python3
"""Summarize fixed GSM8K PoT validation-token NLL across SFT checkpoints."""
import json
import os
from pathlib import Path

root = Path(os.environ.get("NLL_ROOT", "eval_runs/gsm8k-sft-target-alignment/target_nll"))
out = Path(os.environ.get("NLL_REPORT", root / "target_nll_summary.md"))
steps = [("Base EMA", "base"), ("Step 100 EMA", "step100"),
         ("Step 200 EMA", "step200"), ("Step 711 EMA", "step711")]
rows = []
for label, name in steps:
    path = root / f"{name}.json"
    result = json.loads(path.read_text())
    if isinstance(result, list):
        result = result[0]
    rows.append((label, result))

keys = sorted({key for _, row in rows for key in row
               if "nll" in key.lower() or "ppl" in key.lower()})
lines = ["# GSM8K PoT teacher-forced validation loss", "",
         "All checkpoints were evaluated with the same validation split, EMA weights, objective, seed, and batch settings. Lower NLL is better. `code_eos_nll` is the monitored non-padding supervised target region; `nll` follows the configured objective/mask.", ""]
if keys:
    lines += ["| Checkpoint | " + " | ".join(keys) + " |",
              "|---|" + "---:|" * len(keys)]
    for label, row in rows:
        vals = [f"{float(row[k]):.6f}" if k in row else "—" for k in keys]
        lines.append("| " + label + " | " + " | ".join(vals) + " |")
else:
    lines += ["No NLL/PPL fields were present. Inspect the individual result JSON files."]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n")
print(out)
