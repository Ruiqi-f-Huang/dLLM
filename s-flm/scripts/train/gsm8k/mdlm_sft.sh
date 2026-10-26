#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
RUN_ID="$(date +%Y%m%d-%H%M%S)"
python -u -m main --config-name "${SFT_CONFIG_NAME:-sft}" \
  training.finetune_path="${CKPT_PATH:-${REPO_ROOT}/checkpoints/tinygsm/mdlm.ckpt}" \
  data.train_file="${TRAIN_FILE:-${REPO_ROOT}/data/gsm8k_pot/sft_train.jsonl}" \
  data.validation_file="${VALID_FILE:-${REPO_ROOT}/data/gsm8k_pot/sft_validation.jsonl}" \
  data.cache_dir="${CACHE_DIR:-${REPO_ROOT}/data_cache/gsm8k_pot}" \
  hydra.run.dir="${OUTPUT_DIR:-${REPO_ROOT}/outputs/sft/${RUN_ID}}" \
  "$@"
