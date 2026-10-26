#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HUB_OFFLINE=1
ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/s-flm/outputs/selftrain-pilot-20261023-01}"
mkdir -p "$ROOT/acquisition"
for spec in lowtemp:0.1:1 candidate11:0.7:11 candidate22:0.7:22 candidate33:0.7:33 candidate44:0.7:44; do
 IFS=: read -r label temperature seed <<< "$spec"
 run="$ROOT/acquisition/$label"
 if [[ -f "$run/results.json" ]]; then continue; fi
 python -u -m main mode=gsm8k_eval \
  eval.checkpoint_path="$REPO_ROOT/checkpoints/tinygsm/mdlm.ckpt" eval.disable_ema=false lora.enabled=false \
  data=gsm8k-pot data.validation_file="$REPO_ROOT/data/gsm8k_pot_selftrain_pilot/generation_pool.jsonl" \
  data.cache_dir="$REPO_ROOT/data_cache/gsm8k_pot" model=small model.length=512 algo=mdlm sampler=ancestral \
  sampler.temperature="$temperature" sampler.steps=32 seed="$seed" loader.eval_batch_size=32 loader.num_workers=4 \
  trainer.num_nodes=1 trainer.devices=1 gsm8k.output_dir="$run" +wandb.offline=true hydra.run.dir="$run" \
  > "$ROOT/acquisition/$label.log" 2>&1
done
touch "$ROOT/ACQUISITION_COMPLETE"
