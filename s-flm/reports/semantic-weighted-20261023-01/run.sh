#!/usr/bin/env bash
set -euo pipefail
source /root/s-flm/activate.sh
RUN=/root/autodl-tmp/s-flm/outputs/semantic-weighted-20261023-01
SFT_CONFIG_NAME=sft-semantic-weighted OUTPUT_DIR="$RUN/train" bash scripts/train/gsm8k/mdlm_sft.sh wandb.id=sw2o23a1 wandb.name=mdlm-sft-semantic-w2-711 trainer.log_every_n_steps=10 +trainer.enable_progress_bar=false > "$RUN/train.log" 2>&1
CKPT_DIR="$RUN/train/checkpoints" OUTPUT_ROOT="$RUN/eval" MILESTONES="0 100 200 711" bash scripts/sample/tinygsm/mdlm_lora_milestones.sh > "$RUN/eval.log" 2>&1
CKPT_DIR=/root/autodl-tmp/s-flm/outputs/sft/20261022-lora-control-01/checkpoints OUTPUT_ROOT="$RUN/retained-control" MILESTONES="200" bash scripts/sample/tinygsm/mdlm_lora_milestones.sh > "$RUN/retained-control.log" 2>&1
touch "$RUN/COMPLETE"
