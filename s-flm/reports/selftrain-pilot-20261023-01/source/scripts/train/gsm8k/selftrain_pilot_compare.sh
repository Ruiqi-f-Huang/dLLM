#!/usr/bin/env bash
# Run only after candidate filtering and manual review of review32.json.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HUB_OFFLINE=1
ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/s-flm/outputs/selftrain-pilot-20261023-01}"
python - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]); s=json.loads((r/'filter_summary.json').read_text())
assert s['training_eligible'], s
assert (r/'MANUAL_REVIEW_COMPLETE').exists()
PY
for arm in original generated; do
 SFT_CONFIG_NAME=sft-selftrain-pilot \
 TRAIN_FILE="$REPO_ROOT/data/gsm8k_pot_selftrain_pilot/sft_train_${arm}.jsonl" \
 OUTPUT_DIR="$ROOT/train-$arm" \
 bash scripts/train/gsm8k/mdlm_sft.sh \
  wandb.id="stp23w0${arm}" wandb.name="selftrain-${arm}-200" \
  +trainer.enable_progress_bar=false > "$ROOT/train-$arm.log" 2>&1
done
mkdir -p "$ROOT/evaluation"
# Reuse the completed, identically configured Base seed1 validation run.
cp -a /root/autodl-tmp/s-flm/outputs/lowtemp-sft-comparison-20261023-01/base "$ROOT/evaluation/base-seed1"
for label in base original generated; do
 if [[ "$label" == base ]]; then ckpt="$REPO_ROOT/checkpoints/tinygsm/mdlm.ckpt"; lora=false; else ckpt="$ROOT/train-$label/checkpoints/step-000200.ckpt"; lora=true; fi
 for seed in 1 2; do
  if [[ "$label" == base && "$seed" == 1 ]]; then continue; fi
  run="$ROOT/evaluation/$label-seed$seed"
  python -u -m main mode=gsm8k_eval eval.checkpoint_path="$ckpt" eval.disable_ema=false lora.enabled="$lora" \
   data=gsm8k-pot model=small model.length=512 algo=mdlm sampler=ancestral sampler.steps=32 sampler.temperature=0.1 \
   loader.eval_batch_size=32 loader.num_workers=4 trainer.num_nodes=1 trainer.devices=1 seed="$seed" \
   gsm8k.output_dir="$run" +wandb.offline=true hydra.run.dir="$run" > "$ROOT/evaluation-$label-seed$seed.log" 2>&1
 done
done
python scripts/analyze_selftrain_pilot.py "$ROOT" > "$ROOT/comparison.log" 2>&1
touch "$ROOT/COMPLETE"
