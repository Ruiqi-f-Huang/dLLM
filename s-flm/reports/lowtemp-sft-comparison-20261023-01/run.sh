#!/usr/bin/env bash
# Full validation comparison; checkpoint weights and sampling settings are explicit.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HUB_OFFLINE=1
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/s-flm/outputs/lowtemp-sft-comparison-20261023-01}"
WEIGHTED_ROOT="${WEIGHTED_ROOT:-/root/autodl-tmp/s-flm/outputs/semantic-weighted-20261023-01/train/checkpoints}"
CONTROL_CKPT="${CONTROL_CKPT:-/root/autodl-tmp/s-flm/checkpoints/lora-control-step200.ckpt}"
BASE_CKPT="${BASE_CKPT:-$REPO_ROOT/checkpoints/tinygsm/mdlm.ckpt}"
mkdir -p "$OUTPUT_ROOT"
labels=(base uniform200 weighted200 weighted711)
paths=("$BASE_CKPT" "$CONTROL_CKPT" "$WEIGHTED_ROOT/step-000200.ckpt" "$WEIGHTED_ROOT/step-000711.ckpt")
for ckpt in "${paths[@]}"; do test -f "$ckpt"; done
for index in 0 1 2 3; do
  label="${labels[$index]}"
  lora=true
  if [[ "$label" == base ]]; then lora=false; fi
  run="$OUTPUT_ROOT/$label"
  python -u -m main mode=gsm8k_eval \
    eval.checkpoint_path="${paths[$index]}" eval.disable_ema=false lora.enabled="$lora" \
    data=gsm8k-pot data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.validation_file="$REPO_ROOT/data/gsm8k_pot/sft_validation.jsonl" \
    data.cache_dir="$REPO_ROOT/data_cache/gsm8k_pot" \
    model=small model.length=512 algo=mdlm sampler=ancestral \
    sampler.temperature=0.1 sampler.steps=32 \
    loader.eval_batch_size=32 loader.num_workers=4 trainer.num_nodes=1 trainer.devices=1 \
    seed=1 gsm8k.output_dir="$run" +wandb.offline=true hydra.run.dir="$run" \
    > "$OUTPUT_ROOT/$label.log" 2>&1
done
python scripts/compare_lowtemp_sft.py "$OUTPUT_ROOT" > "$OUTPUT_ROOT/comparison.log" 2>&1
touch "$OUTPUT_ROOT/COMPLETE"
