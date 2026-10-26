#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RUN_ID="${RUN_ID:-20261022-lora-result-cot-comments-711-01}"
TRAIN_ROOT="${TRAIN_ROOT:-/root/autodl-tmp/s-flm/outputs/sft/${RUN_ID}}"
RESUME_CKPT="${RESUME_CKPT:-/root/autodl-tmp/s-flm/outputs/sft/20261022-lora-result-cot-comments-pilot-01/checkpoints/step-000200.ckpt}"
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

for f in "${RESUME_CKPT}" "${TRAIN_FILE}" "${VALID_FILE}"; do
  [[ -s "${f}" ]] || { echo "Required input missing: ${f}" >&2; exit 2; }
done
[[ ! -e "${TRAIN_ROOT}/COMPLETE" ]] || {
  echo "Run already has a COMPLETE marker: ${TRAIN_ROOT}" >&2
  exit 3
}
mkdir -p "${TRAIN_ROOT}" "${CACHE_DIR}"

python - "${RESUME_CKPT}" <<'PY'
import sys
import torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert ckpt.get("global_step") == 200, ckpt.get("global_step")
assert ckpt.get("optimizer_states"), "optimizer state missing"
assert ckpt.get("lr_schedulers"), "scheduler state missing"
assert ckpt.get("ema"), "EMA state missing"
print("Verified resumable step-200 checkpoint with optimizer, scheduler, and EMA.")
PY

echo "Resuming the annotated GSM8K PoT run from step 200 through step 711."
python -u -m main --config-name sft-a800-lora-result-cot-comments-pilot \
  data.train_file="${TRAIN_FILE}" \
  data.validation_file="${VALID_FILE}" \
  data.cache_dir="${CACHE_DIR}" \
  data.tokenizer_name_or_path="${TOKENIZER_PATH}" \
  optim.lr=1e-5 \
  trainer.max_steps=711 \
  callbacks.milestone_checkpoint.milestones='[100,200,711]' \
  checkpointing.save_dir="${TRAIN_ROOT}" \
  checkpointing.resume_from_ckpt=true \
  checkpointing.resume_ckpt_path="${RESUME_CKPT}" \
  hydra.run.dir="${TRAIN_ROOT}" \
  wandb.offline=true \
  wandb.name="${RUN_ID}" \
  wandb.save_dir="${TRAIN_ROOT}" \
  2>&1 | tee "${TRAIN_ROOT}/training.log"

FINAL_CKPT="${TRAIN_ROOT}/checkpoints/step-000711.ckpt"
[[ -s "${FINAL_CKPT}" ]] || { echo "Missing step-711 checkpoint" >&2; exit 4; }
python - "${FINAL_CKPT}" <<'PY'
import sys
import torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert ckpt.get("global_step") == 711, ckpt.get("global_step")
print("Verified final checkpoint at optimizer step 711.")
PY
printf 'experiment_complete=%s\n' "$(date -Is)" > "${TRAIN_ROOT}/COMPLETE"
echo "Annotated GSM8K PoT training complete: ${RUN_ID}"
