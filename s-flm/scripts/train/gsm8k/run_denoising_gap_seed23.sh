#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RUN_ID="${RUN_ID:-20261022-denoising-gap-seed23}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/s-flm/outputs/eval/${RUN_ID}}"
BASE_CKPT="${BASE_CKPT:-/root/autodl-tmp/s-flm/checkpoints/tinygsm/mdlm.ckpt}"
SFT_ROOT="${SFT_ROOT:-/root/autodl-tmp/s-flm/outputs/sft/20261022-target-learning-lora-lr1e5/checkpoints}"
VALID_FILE="${VALID_FILE:-${REPO_ROOT}/data/gsm8k_pot_result/sft_validation.jsonl}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/root/autodl-tmp/s-flm/cache/huggingface/hub/models--HuggingFaceTB--SmolLM-135M/snapshots/1d461723eec654e65efdc40cf49301c89c0c92f4}"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export DIAG_OUT="${OUTPUT_ROOT}"
export DIAG_SEEDS="2,3"
export DIAG_CHECKPOINTS="base,step711"
export DIAG_SKIP_MASK_SWEEP=1
export DIAG_GENERATION_SUBSET=128
export DIAG_BATCH_SIZE=8
export BASE_CKPT
export STEP711_CKPT="${SFT_ROOT}/step-000711.ckpt"

if [[ -f /root/venvs/s-flm/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /root/venvs/s-flm/bin/activate
fi

cd "${REPO_ROOT}"
for f in "${BASE_CKPT}" "${STEP711_CKPT}" "${VALID_FILE}"; do
  [[ -s "${f}" ]] || { echo "Required diagnostic input missing: ${f}" >&2; exit 2; }
done

mkdir -p "${OUTPUT_ROOT}"
python -u scripts/eval_mdlm_denoising_gap.py \
  data.validation_file="${VALID_FILE}" \
  data.cache_dir=/root/autodl-tmp/s-flm/data_cache/gsm8k_pot_result \
  data.tokenizer_name_or_path="${TOKENIZER_PATH}" \
  data.tokenizer_revision=1d461723eec654e65efdc40cf49301c89c0c92f4 \
  model.length=512 \
  loader.eval_batch_size=8 \
  trainer.devices=1 trainer.num_nodes=1 \
  hydra.run.dir="${OUTPUT_ROOT}/hydra" \
  2>&1 | tee "${OUTPUT_ROOT}/diagnostic.log"

[[ -s "${OUTPUT_ROOT}/COMPLETE" ]] || {
  echo "Diagnostic did not write COMPLETE marker." >&2
  exit 3
}
echo "Generation multi-seed diagnostic completed: ${OUTPUT_ROOT}"
