#!/usr/bin/env bash
set -euo pipefail

# One unattended job: train LoRA SFT, then measure fixed-target NLL and
# free-generation/AST alignment at Base, 100, 200, and 711 optimizer steps.
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RUN_ID="${RUN_ID:-20261022-target-learning-lora-lr1e5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/s-flm/outputs/sft/${RUN_ID}}"
EVAL_ROOT="${EVAL_ROOT:-/root/autodl-tmp/s-flm/outputs/eval/${RUN_ID}}"
BASE_CKPT="${BASE_CKPT:-${REPO_ROOT}/checkpoints/tinygsm/mdlm.ckpt}"
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/gsm8k_pot_result/sft_train.jsonl}"
VALID_FILE="${VALID_FILE:-${REPO_ROOT}/data/gsm8k_pot_result/sft_validation.jsonl}"
TRAIN_SUBSET_FILE="${TRAIN_SUBSET_FILE:-${REPO_ROOT}/data/gsm8k_pot_result/diagnostics/train_eval_subset_seed20261022_n256.jsonl}"
CACHE_DIR="${CACHE_DIR:-/root/autodl-tmp/s-flm/data_cache/gsm8k_pot_result}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/root/autodl-tmp/s-flm/cache/huggingface/hub/models--HuggingFaceTB--SmolLM-135M/snapshots/1d461723eec654e65efdc40cf49301c89c0c92f4}"
SAMPLING_STEPS="${SAMPLING_STEPS:-32}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_NLL="${SKIP_NLL:-0}"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
if [[ -f /root/venvs/s-flm/bin/activate ]]; then
  # A nohup/SSH-launched process does not inherit the interactive shell's venv.
  # shellcheck disable=SC1091
  source /root/venvs/s-flm/bin/activate
fi
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${EVAL_ROOT}"

for f in "${BASE_CKPT}" "${TRAIN_FILE}" "${VALID_FILE}" "${TRAIN_SUBSET_FILE}"; do
  [[ -f "${f}" ]] || { echo "Required input missing: ${f}" >&2; exit 2; }
done

if [[ "${SKIP_TRAIN}" != 1 ]]; then
  echo "[1/4] Starting one-epoch LoRA SFT. run_id=${RUN_ID}"
  python -u -m main --config-name sft-a800-lora-result-lr1e5 \
    data.train_file="${TRAIN_FILE}" \
    data.validation_file="${VALID_FILE}" \
    data.cache_dir="${CACHE_DIR}" \
    data.tokenizer_name_or_path="${TOKENIZER_PATH}" \
    optim.lr=1e-5 \
    callbacks.milestone_checkpoint.milestones='[100,200,711]' \
    checkpointing.save_dir="${OUTPUT_ROOT}" \
    hydra.run.dir="${OUTPUT_ROOT}" \
    wandb.offline=true \
    wandb.name="${RUN_ID}" \
    wandb.save_dir="${OUTPUT_ROOT}" \
    2>&1 | tee "${OUTPUT_ROOT}/training.log"
else
  echo "[1/4] SKIP_TRAIN=1; reusing existing checkpoints."
fi

CKPT_DIR="${OUTPUT_ROOT}/checkpoints"
for step in 100 200 711; do
  printf -v checkpoint 'step-%06d.ckpt' "${step}"
  [[ -s "${CKPT_DIR}/${checkpoint}" ]] || { echo "Missing milestone checkpoint ${checkpoint}" >&2; exit 3; }
done
echo "[2/4] Training completed; all requested checkpoints exist."

eval_nll() {
  local label="$1" checkpoint="$2" lora="$3"
  local result="${EVAL_ROOT}/target_nll/${label}.json"
  mkdir -p "${EVAL_ROOT}/target_nll"
  python -u -m main --config-name sft-a800-lora-result-lr1e5 \
    mode=ppl_eval eval.checkpoint_path="${checkpoint}" eval.disable_ema=false \
    eval.results_json_path="${result}" lora.enabled="${lora}" \
    data.train_file="${TRAIN_FILE}" data.validation_file="${VALID_FILE}" \
    data.cache_dir="${CACHE_DIR}" data.tokenizer_name_or_path="${TOKENIZER_PATH}" \
    model.length=512 loader.eval_batch_size=8 loader.eval_global_batch_size=8 \
    trainer.devices=1 trainer.num_nodes=1 seed=1 \
    sft_monitoring=true wandb.offline=true \
    hydra.run.dir="${EVAL_ROOT}/target_nll/${label}-run" \
    2>&1 | tee "${EVAL_ROOT}/target_nll/${label}.log"
  [[ -s "${result}" ]] || { echo "NLL evaluation did not produce ${result}" >&2; exit 4; }
}

if [[ "${SKIP_NLL}" != 1 ]]; then
  echo "[3/4] Measuring fixed-target validation NLL for Base/100/200/711."
  eval_nll base "${BASE_CKPT}" false
  eval_nll step100 "${CKPT_DIR}/step-000100.ckpt" true
  eval_nll step200 "${CKPT_DIR}/step-000200.ckpt" true
  eval_nll step711 "${CKPT_DIR}/step-000711.ckpt" true
  NLL_ROOT="${EVAL_ROOT}/target_nll" \
  NLL_REPORT="${EVAL_ROOT}/target_nll/target_nll_summary.md" \
    python "${REPO_ROOT}/scripts/analyze_sft_target_nll.py"
else
  echo "[3/4] SKIP_NLL=1; reusing existing NLL results."
fi

eval_generation() {
  local split="$1" label="$2" checkpoint="$3" lora="$4" input_file="$5"
  local run_dir="${EVAL_ROOT}/${split}/${label}-steps${SAMPLING_STEPS}-seed1"
  mkdir -p "${run_dir}"
  python -u -m main \
    mode=gsm8k_eval eval.checkpoint_path="${checkpoint}" eval.disable_ema=false \
    lora.enabled="${lora}" data=gsm8k-pot-result \
    data.train_file="${TRAIN_FILE}" data.validation_file="${input_file}" \
    data.cache_dir="${CACHE_DIR}" data.tokenizer_name_or_path="${TOKENIZER_PATH}" \
    model=small model.length=512 algo=mdlm sampler=ancestral \
    sampler.steps="${SAMPLING_STEPS}" loader.eval_batch_size=32 \
    loader.eval_global_batch_size=32 loader.num_workers=4 \
    trainer.num_nodes=1 trainer.devices=1 seed=1 \
    gsm8k.output_dir="${run_dir}" ++wandb.offline=true \
    hydra.run.dir="${run_dir}" \
    2>&1 | tee "${run_dir}/eval.log"
  [[ -s "${run_dir}/results.json" ]] || { echo "Generation evaluation failed: ${run_dir}" >&2; exit 5; }
}

echo "[4/4] Sampling Base/100/200/711 on validation and the fixed 256-row train probe."
for spec in \
  "base|${BASE_CKPT}|false" \
  "step-100|${CKPT_DIR}/step-000100.ckpt|true" \
  "step-200|${CKPT_DIR}/step-000200.ckpt|true" \
  "step-711|${CKPT_DIR}/step-000711.ckpt|true"; do
  IFS='|' read -r label checkpoint lora <<< "${spec}"
  eval_generation validation "${label}-ema" "${checkpoint}" "${lora}" "${VALID_FILE}"
  eval_generation train-subset "${label}-ema" "${checkpoint}" "${lora}" "${TRAIN_SUBSET_FILE}"
done

EVAL_ROOT="${EVAL_ROOT}" \
ALIGNMENT_CSV="${EVAL_ROOT}/target_alignment.csv" \
ALIGNMENT_MD="${EVAL_ROOT}/target_alignment.md" \
  python "${REPO_ROOT}/scripts/analyze_sft_target_alignment.py"

printf 'experiment_complete=%s\n' "$(date -Is)" > "${EVAL_ROOT}/COMPLETE"
echo "All training and diagnostics completed. Results: ${EVAL_ROOT}"
