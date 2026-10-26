#!/usr/bin/env bash
set -euo pipefail

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BASE_CKPT="${BASE_CKPT:-${REPO_ROOT}/checkpoints/tinygsm/mdlm.ckpt}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/outputs/sft/base-aligned/checkpoints}"
VALID_FILE="${VALID_FILE:-${REPO_ROOT}/data/gsm8k_pot/sft_validation.jsonl}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache/gsm8k_pot}"
TOKENIZER_PATH="${TOKENIZER_PATH:-HuggingFaceTB/SmolLM-135M}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/eval_runs/gsm8k-sft-validation/base-aligned}"
SAMPLING_STEPS="${SAMPLING_STEPS:-32}"

cd "${REPO_ROOT}"

for TRAIN_STEP in 0 100 200 711; do
  if [[ "${TRAIN_STEP}" == 0 ]]; then
    CHECKPOINT="${BASE_CKPT}"
  else
    printf -v STEP_NAME 'step-%06d.ckpt' "${TRAIN_STEP}"
    CHECKPOINT="${CKPT_DIR}/${STEP_NAME}"
  fi
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
  fi

  RUN_DIR="${OUTPUT_ROOT}/step-${TRAIN_STEP}-ema-steps${SAMPLING_STEPS}-seed1"
  python -u -m main \
    mode=gsm8k_eval \
    eval.checkpoint_path="${CHECKPOINT}" \
    eval.disable_ema=false \
    data=gsm8k-pot \
    data.tokenizer_name_or_path="${TOKENIZER_PATH}" \
    data.cache_dir="${CACHE_DIR}" \
    data.validation_file="${VALID_FILE}" \
    model=small \
    model.length=512 \
    algo=mdlm \
    sampler=ancestral \
    sampler.steps="${SAMPLING_STEPS}" \
    loader.eval_batch_size=32 \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    seed=1 \
    gsm8k.output_dir="${RUN_DIR}" \
    +wandb.offline=true \
    hydra.run.dir="${RUN_DIR}"
done

python scripts/sample/tinygsm/summarize_sft_milestones.py "${OUTPUT_ROOT}"

