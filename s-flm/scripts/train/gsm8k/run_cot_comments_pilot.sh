#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RUN_ID="${RUN_ID:-20261022-lora-result-cot-comments-pilot-01}"
TRAIN_ROOT="${TRAIN_ROOT:-/root/autodl-tmp/s-flm/outputs/sft/${RUN_ID}}"
EVAL_ROOT="${EVAL_ROOT:-/root/autodl-tmp/s-flm/outputs/eval/${RUN_ID}}"
BASE_CKPT="${BASE_CKPT:-${REPO_ROOT}/checkpoints/tinygsm/mdlm.ckpt}"
CONTROL_CKPT="${CONTROL_CKPT:-/root/autodl-tmp/s-flm/outputs/sft/20261022-lora-result-lr1e5-01/checkpoints/step-000200.ckpt}"
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/gsm8k_pot_result_cot_comments_512/sft_train.jsonl}"
VALID_FILE="${VALID_FILE:-${REPO_ROOT}/data/gsm8k_pot_result_cot_comments_512/sft_validation.jsonl}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/root/autodl-tmp/s-flm/cache/huggingface/hub/models--HuggingFaceTB--SmolLM-135M/snapshots/1d461723eec654e65efdc40cf49301c89c0c92f4}"
CACHE_DIR="${CACHE_DIR:-/root/autodl-tmp/s-flm/data_cache/gsm8k_pot_result_cot_comments_512}"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
if [[ -f /root/venvs/s-flm/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /root/venvs/s-flm/bin/activate
fi
cd "${REPO_ROOT}"
mkdir -p "${TRAIN_ROOT}" "${EVAL_ROOT}" "${CACHE_DIR}"

for f in "${BASE_CKPT}" "${CONTROL_CKPT}" "${TRAIN_FILE}" "${VALID_FILE}"; do
  [[ -s "${f}" ]] || { echo "Required input missing: ${f}" >&2; exit 2; }
done
[[ ! -e "${TRAIN_ROOT}/COMPLETE" ]] || {
  echo "Run already has a COMPLETE marker: ${TRAIN_ROOT}" >&2
  exit 3
}

echo "[1/3] Rechecking tokenizer lengths (must all fit 512 tokens)."
python scripts/data/validate_gsm8k_pot_tokens.py \
  --data-dir "$(dirname "${TRAIN_FILE}")" \
  --tokenizer "${TOKENIZER_PATH}" --separator '\n' --max-length 512 \
  --output "$(dirname "${TRAIN_FILE}")/tokenization_report.json"

echo "[2/3] Training the reference-comment LoRA pilot for 200 steps."
python -u -m main --config-name sft-a800-lora-result-cot-comments-pilot \
  data.train_file="${TRAIN_FILE}" \
  data.validation_file="${VALID_FILE}" \
  data.cache_dir="${CACHE_DIR}" \
  data.tokenizer_name_or_path="${TOKENIZER_PATH}" \
  optim.lr=1e-5 \
  trainer.max_steps=200 \
  callbacks.milestone_checkpoint.milestones='[100,200]' \
  checkpointing.save_dir="${TRAIN_ROOT}" \
  hydra.run.dir="${TRAIN_ROOT}" \
  wandb.offline=true \
  wandb.name="${RUN_ID}" \
  wandb.save_dir="${TRAIN_ROOT}" \
  2>&1 | tee "${TRAIN_ROOT}/training.log"

NEW_CKPT="${TRAIN_ROOT}/checkpoints/step-000200.ckpt"
[[ -s "${NEW_CKPT}" ]] || { echo "Missing step-200 checkpoint" >&2; exit 4; }

echo "[3/3] Comparing Base, code-only SFT200 and CoT-comment SFT200."
export DIAG_OUT="${EVAL_ROOT}"
export DIAG_SEEDS="1,2"
export DIAG_SAMPLE_STEPS="128"
export DIAG_CHECKPOINTS="base,step200,cotcomments200"
export DIAG_SKIP_MASK_SWEEP=1
export DIAG_GENERATION_SUBSET=64
export DIAG_BATCH_SIZE=8
export BASE_CKPT
export STEP200_CKPT="${CONTROL_CKPT}"
export COTCOMMENTS_CKPT="${NEW_CKPT}"
python -u scripts/eval_mdlm_denoising_gap.py \
  data.validation_file="${VALID_FILE}" \
  data.cache_dir="${CACHE_DIR}" \
  data.tokenizer_name_or_path="${TOKENIZER_PATH}" \
  data.tokenizer_revision=1d461723eec654e65efdc40cf49301c89c0c92f4 \
  model.length=512 loader.eval_batch_size=8 \
  trainer.devices=1 trainer.num_nodes=1 \
  hydra.run.dir="${EVAL_ROOT}/hydra" \
  2>&1 | tee "${EVAL_ROOT}/diagnostic.log"

[[ -s "${EVAL_ROOT}/COMPLETE" ]] || {
  echo "Diagnostic did not write COMPLETE marker." >&2
  exit 5
}
printf 'experiment_complete=%s\n' "$(date -Is)" > "${TRAIN_ROOT}/COMPLETE"
echo "Pilot training and evaluation completed: ${RUN_ID}"
