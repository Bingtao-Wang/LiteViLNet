#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 GPU_ID CONFIG:SEED [CONFIG:SEED ...]" >&2
  exit 2
fi

GPU_ID="$1"
shift

: "${LITEVILNET_DATA_ROOT:?Set LITEVILNET_DATA_ROOT to the KITTI Road data root}"
DATA_ROOT="${LITEVILNET_DATA_ROOT}"
OUTPUT_ROOT="${LITEVILNET_REVISION_OUTPUT:-runs/revision_1/kitti_ablation}"
TRAIN_SPLIT="${LITEVILNET_KITTI_TRAIN_SPLIT:-configs/splits/kitti_road/train.txt}"
VAL_SPLIT="${LITEVILNET_KITTI_VAL_SPLIT:-configs/splits/kitti_road/val.txt}"

for SPEC in "$@"; do
  CONFIG="${SPEC%%:*}"
  SEED="${SPEC##*:}"
  echo "[$(date --iso-8601=seconds)] starting ${CONFIG} seed=${SEED} on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHONPATH=. \
  NO_ALBUMENTATIONS_UPDATE=1 \
  conda run -n litevilnet_ral python -m tools.train_ablation \
    --config "${CONFIG}" \
    --data_root "${DATA_ROOT}" \
    --train_split_file "${TRAIN_SPLIT}" \
    --val_split_file "${VAL_SPLIT}" \
    --img_h 384 \
    --img_w 1248 \
    --epochs 150 \
    --batch_size 2 \
    --accumulate_grad_batches 8 \
    --lr 2e-4 \
    --weight_decay 5e-4 \
    --patience 40 \
    --amp \
    --num_workers 4 \
    --seed "${SEED}" \
    --drop_last \
    --deterministic \
    --skip_train_metrics \
    --save_dir "${OUTPUT_ROOT}" \
    --log_dir "${OUTPUT_ROOT}/logs"
  echo "[$(date --iso-8601=seconds)] completed ${CONFIG} seed=${SEED}"
done
