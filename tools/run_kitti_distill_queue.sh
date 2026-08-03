#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 GPU_ID SEED [SEED ...]" >&2
  exit 2
fi

GPU_ID="$1"
shift

: "${LITEVILNET_KITTI_ROOT:?Set LITEVILNET_KITTI_ROOT to the KITTI Road data root}"
: "${LITEVILNET_KITTI_TEACHER:?Set LITEVILNET_KITTI_TEACHER to the full-model checkpoint}"
DATA_ROOT="${LITEVILNET_KITTI_ROOT}"
TRAIN_SPLIT="${LITEVILNET_KITTI_TRAIN_SPLIT:-configs/splits/kitti_road/stratified_seed20260723/train.txt}"
VAL_SPLIT="${LITEVILNET_KITTI_VAL_SPLIT:-configs/splits/kitti_road/stratified_seed20260723/val.txt}"
TEACHER="${LITEVILNET_KITTI_TEACHER}"
OUTPUT_ROOT="${LITEVILNET_KITTI_DISTILL_OUTPUT:-runs/revision_1/kitti_distill_edge}"

for SEED in "$@"; do
  echo "[$(date --iso-8601=seconds)] starting KITTI KD student seed=${SEED} on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHONPATH=. \
  NO_ALBUMENTATIONS_UPDATE=1 \
  conda run --no-capture-output -n litevilnet_ral python -m tools.train_distill_edge \
    --data_root "${DATA_ROOT}" \
    --train_split_file "${TRAIN_SPLIT}" \
    --val_split_file "${VAL_SPLIT}" \
    --teacher_checkpoint "${TEACHER}" \
    --student_preset litevilnet_edge \
    --save_dir "${OUTPUT_ROOT}" \
    --log_dir "${OUTPUT_ROOT}/logs" \
    --img_h 384 \
    --img_w 1248 \
    --batch_size 2 \
    --accumulate_grad_batches 8 \
    --num_workers 4 \
    --epochs 150 \
    --lr 2e-4 \
    --weight_decay 5e-4 \
    --amp \
    --patience 40 \
    --seed "${SEED}" \
    --deterministic \
    --drop_last \
    --skip_train_metrics
  echo "[$(date --iso-8601=seconds)] completed KITTI KD student seed=${SEED}"
done
