#!/bin/bash
set -euo pipefail
source /root/s-flm/activate.sh
python -m pip check > /root/setup-logs/a800-pip-check.log
python /root/setup-logs/test-a800-kernels.py > /root/setup-logs/a800-kernel-test.log 2>&1
python -m unittest discover -s tests -p 'test_sft*.py' -v > /root/setup-logs/a800-sft-tests.log 2>&1
bash scripts/train/gsm8k/mdlm_sft.sh --cfg job --resolve > /root/setup-logs/a800-sft-resolved.yaml
OUTPUT_DIR=/root/autodl-tmp/s-flm/outputs/a800-env-smoke bash scripts/train/gsm8k/mdlm_sft.sh wandb.offline=true trainer.max_steps=2 trainer.limit_train_batches=2 trainer.limit_val_batches=2 trainer.log_every_n_steps=1 +trainer.enable_progress_bar=false > /root/setup-logs/a800-sft-smoke.log 2>&1
