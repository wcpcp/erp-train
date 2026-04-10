#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
source "${ROOT_DIR}/scripts/common_dataset.sh"

TRAIN_DATA="${TRAIN_DATA:-${ROOT_DIR}/examples/data/pano_erp_sft_sample.jsonl}"
VAL_DATA="${VAL_DATA:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/qwen2_5_vl_3b_erp_lora}"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-pano_qwen2_5_vl}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"
export PANO_ERP_HIDDEN_DIM="${PANO_ERP_HIDDEN_DIM:-512}"
export PANO_ERP_GATE_INIT="${PANO_ERP_GATE_INIT:-0.01}"
export PANO_ERP_POS_MODE="${PANO_ERP_POS_MODE:-paper}"
export PANO_ERP_STAGE="${PANO_ERP_STAGE:-output}"
export PANO_ERP_TARGET="${PANO_ERP_TARGET:-both}"

ERP_MODULES_TO_SAVE=()
[[ ",${PANO_ERP_STAGE}," == *",patch,"* ]] && ERP_MODULES_TO_SAVE+=(erp_patch_adapter)
[[ ",${PANO_ERP_STAGE}," == *",merger,"* ]] && ERP_MODULES_TO_SAVE+=(erp_merger_adapter)
[[ ",${PANO_ERP_STAGE}," == *",output,"* ]] && ERP_MODULES_TO_SAVE+=(erp_output_adapter)
ERP_MODULES_TO_SAVE_CSV="$(IFS=,; echo "${ERP_MODULES_TO_SAVE[*]}")"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

TRAIN_DATA="$(normalize_dataset_for_swift "${TRAIN_DATA}" train)"
if [[ -n "${VAL_DATA}" ]]; then
  VAL_DATA="$(normalize_dataset_for_swift "${VAL_DATA}" val)"
fi

DATA_ARGS=(--dataset "${TRAIN_DATA}")
if [[ -n "${VAL_DATA}" ]]; then
  DATA_ARGS+=(--val_dataset "${VAL_DATA}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
swift sft \
  --model "${MODEL}" \
  --model_type "${MODEL_TYPE}" \
  --external_plugins "${ROOT_DIR}/src/pano_qwen_erp/register.py" \
  "${DATA_ARGS[@]}" \
  --load_from_cache_file true \
  --split_dataset_ratio 0.0 \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-2}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
  --attn_impl flash_attn \
  --padding_free true \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --lora_rank "${LORA_RANK:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --target_modules all-linear \
  --modules_to_save "${ERP_MODULES_TO_SAVE_CSV}" \
  --freeze_vit true \
  --freeze_aligner true \
  --packing true \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing false \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --eval_steps "${EVAL_STEPS:-200}" \
  --save_steps "${SAVE_STEPS:-200}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --max_length "${MAX_LENGTH:-4096}" \
  --output_dir "${OUTPUT_DIR}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --deepspeed "${DEEPSPEED_CONFIG:-zero2}" \
  --dataset_num_proc "${DATASET_NUM_PROC:-4}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}"
