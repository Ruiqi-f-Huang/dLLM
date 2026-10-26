#!/usr/bin/env bash
set -euo pipefail

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BASE_CKPT="${BASE_CKPT:-${REPO_ROOT}/checkpoints/tinygsm/mdlm.ckpt}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/outputs/sft/lora-result-lr1e5/checkpoints}"
VALID_FILE="${VALID_FILE:-${REPO_ROOT}/data/gsm8k_pot_result/sft_validation.jsonl}"
TRAIN_SUBSET_FILE="${TRAIN_SUBSET_FILE:-${REPO_ROOT}/data/gsm8k_pot_result/diagnostics/train_eval_subset_seed20261022_n256.jsonl}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache/gsm8k_pot_result}"
TOKENIZER_PATH="${TOKENIZER_PATH:-HuggingFaceTB/SmolLM-135M}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/eval/lora-result-lr1e5-diagnostics}"
SAMPLING_STEPS="${SAMPLING_STEPS:-32}"

cd "${REPO_ROOT}"

eval_one() {
  local split="$1"
  local label="$2"
  local checkpoint="$3"
  local lora_enabled="$4"
  local disable_ema="$5"
  local input_file="$6"
  local run_dir="${OUTPUT_ROOT}/${split}/${label}-steps${SAMPLING_STEPS}-seed1"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  python -u -m main \
    mode=gsm8k_eval \
    eval.checkpoint_path="${checkpoint}" \
    eval.disable_ema="${disable_ema}" \
    lora.enabled="${lora_enabled}" \
    data=gsm8k-pot-result \
    data.tokenizer_name_or_path="${TOKENIZER_PATH}" \
    data.cache_dir="${CACHE_DIR}" \
    data.validation_file="${input_file}" \
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
    gsm8k.output_dir="${run_dir}" \
    +wandb.offline=true \
    hydra.run.dir="${run_dir}"
}

STEP200="${CKPT_DIR}/step-000200.ckpt"
STEP711="${CKPT_DIR}/step-000711.ckpt"

# Compare raw and EMA SFT weights on the unchanged validation split.
eval_one validation step-200-ema "${STEP200}" true false "${VALID_FILE}"
eval_one validation step-200-raw "${STEP200}" true true "${VALID_FILE}"
eval_one validation step-711-ema "${STEP711}" true false "${VALID_FILE}"
eval_one validation step-711-raw "${STEP711}" true true "${VALID_FILE}"

# Compare Base and SFT fit on the same deterministic 256 training examples.
eval_one train-subset base-ema "${BASE_CKPT}" false false "${TRAIN_SUBSET_FILE}"
eval_one train-subset step-200-ema "${STEP200}" true false "${TRAIN_SUBSET_FILE}"
eval_one train-subset step-200-raw "${STEP200}" true true "${TRAIN_SUBSET_FILE}"
eval_one train-subset step-711-ema "${STEP711}" true false "${TRAIN_SUBSET_FILE}"
eval_one train-subset step-711-raw "${STEP711}" true true "${TRAIN_SUBSET_FILE}"
