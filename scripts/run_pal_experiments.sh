#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="crosscbr"
GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-SteamDebug}"
WANDB_PROJECT="${WANDB_PROJECT:-BT4222-PAL}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi

echo "[run_pal_experiments] $(date '+%Y-%m-%d %H:%M:%S') script_dir=${SCRIPT_DIR}"
echo "[run_pal_experiments] $(date '+%Y-%m-%d %H:%M:%S') dataset=${DATASET} gpu=${GPU_ID} wandb_project=${WANDB_PROJECT}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
echo "[run_pal_experiments] $(date '+%Y-%m-%d %H:%M:%S') activated_conda_env=${CONDA_DEFAULT_ENV}"
echo "[run_pal_experiments] $(date '+%Y-%m-%d %H:%M:%S') python=$(command -v python)"
python -V

cd "${SCRIPT_DIR}"

wandb_args=("--wandb-project" "${WANDB_PROJECT}")
if [[ -n "${WANDB_ENTITY}" ]]; then
  wandb_args+=("--wandb-entity" "${WANDB_ENTITY}")
fi

run_baseline() {
  echo "[run_pal_experiments] $(date '+%Y-%m-%d %H:%M:%S') START baseline"
  python -u train.py -g "${GPU_ID}" -m PAL -d "${DATASET}" -i step0_baseline --attention-type none --fusion-type none "${wandb_args[@]}"
  echo "[run_pal_experiments] $(date '+%Y-%m-%d %H:%M:%S') END baseline"
}

run_global_attention() {
  echo "[run_pal_experiments] $(date '+%Y-%m-%d %H:%M:%S') START global_attention"
  python -u train.py -g "${GPU_ID}" -m PAL -d "${DATASET}" -i step1a_global_attention --attention-type global --fusion-type none "${wandb_args[@]}"
  echo "[run_pal_experiments] $(date '+%Y-%m-%d %H:%M:%S') END global_attention"
}

run_user_attention() {
  echo "[run_pal_experiments] $(date '+%Y-%m-%d %H:%M:%S') START user_attention"
  python -u train.py -g "${GPU_ID}" -m PAL -d "${DATASET}" -i step1b_user_attention --attention-type user --fusion-type none "${wandb_args[@]}"
  echo "[run_pal_experiments] $(date '+%Y-%m-%d %H:%M:%S') END user_attention"
}

run_full_model() {
  echo "[run_pal_experiments] $(date '+%Y-%m-%d %H:%M:%S') START full_model"
  python -u train.py -g "${GPU_ID}" -m PAL -d "${DATASET}" -i step2_full_model --attention-type user --fusion-type user "${wandb_args[@]}"
  echo "[run_pal_experiments] $(date '+%Y-%m-%d %H:%M:%S') END full_model"
}

export_explanations() {
  local checkpoint_name="${1:-}"
  if [[ -z "${checkpoint_name}" ]]; then
    echo "usage: $0 explain <checkpoint_name>"
    echo "example: $0 explain step2_full_model_ED_1_Neg_1_64_0.001_0.0001_16_0.1_0.1_0.1_1_0.04_0.25_attn_user_fusion_user"
    exit 1
  fi

  python export_explanations.py \
    --checkpoint "checkpoints/${DATASET}/PAL/model/${checkpoint_name}" \
    --conf "checkpoints/${DATASET}/PAL/conf/${checkpoint_name}" \
    --topn 5 \
    --top-items 3 \
    --split test
}

run_all() {
  echo "[run_pal_experiments] $(date '+%Y-%m-%d %H:%M:%S') START run_all"
  run_baseline
  run_global_attention
  run_user_attention
  run_full_model
  echo "[run_pal_experiments] $(date '+%Y-%m-%d %H:%M:%S') END run_all"
}

ACTION="${1:-all}"

case "${ACTION}" in
  baseline)
    run_baseline
    ;;
  global_attention)
    run_global_attention
    ;;
  user_attention)
    run_user_attention
    ;;
  full_model)
    run_full_model
    ;;
  all)
    run_all
    ;;
  explain)
    export_explanations "${2:-}"
    ;;
  *)
    echo "unknown action: ${ACTION}"
    echo "available actions: baseline, global_attention, user_attention, full_model, all, explain"
    exit 1
    ;;
esac
