#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 GPU_ID CONFIG:SEED [CONFIG:SEED ...]" >&2
  exit 2
fi

GPU_ID="$1"
shift

DATA_ROOT="${LITEVILNET_ORFD_ROOT:-/data/Database/Research04-LiteViLNet/datasets/ORFD/extracted/Final_Dataset}"
OUTPUT_ROOT="${LITEVILNET_ORFD_OUTPUT:-/data/Database/Research04-LiteViLNet/revision_1_runs/orfd_ablation}"
MAX_ATTEMPTS="${LITEVILNET_ORFD_MAX_ATTEMPTS:-3}"

if ! [[ "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "LITEVILNET_ORFD_MAX_ATTEMPTS must be a positive integer" >&2
  exit 2
fi

for SPEC in "$@"; do
  CONFIG="${SPEC%%:*}"
  SEED="${SPEC##*:}"
  ATTEMPT=1
  while true; do
    echo "[$(date --iso-8601=seconds)] starting ORFD ${CONFIG} seed=${SEED} on GPU ${GPU_ID} (attempt ${ATTEMPT}/${MAX_ATTEMPTS})"
    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONPATH=. \
    NO_ALBUMENTATIONS_UPDATE=1 \
    conda run --no-capture-output -n litevilnet_ral python -m tools.train_ablation \
      --dataset orfd \
      --config "${CONFIG}" \
      --data_root "${DATA_ROOT}" \
      --img_h 704 \
      --img_w 1280 \
      --epochs 30 \
      --batch_size 8 \
      --accumulate_grad_batches 1 \
      --lr 2e-4 \
      --weight_decay 5e-4 \
      --patience 10 \
      --amp \
      --num_workers 8 \
      --seed "${SEED}" \
      --drop_last \
      --deterministic \
      --skip_train_metrics \
      --save_dir "${OUTPUT_ROOT}" \
      --log_dir "${OUTPUT_ROOT}/logs"
    STATUS=$?
    set -e
    if [[ ${STATUS} -eq 0 ]]; then
      break
    fi
    echo "[$(date --iso-8601=seconds)] ORFD ${CONFIG} seed=${SEED} failed with exit ${STATUS}" >&2
    if [[ ${ATTEMPT} -ge ${MAX_ATTEMPTS} ]]; then
      echo "[$(date --iso-8601=seconds)] giving up after ${MAX_ATTEMPTS} attempts" >&2
      exit "${STATUS}"
    fi
    ATTEMPT=$((ATTEMPT + 1))
  done
  echo "[$(date --iso-8601=seconds)] completed ORFD ${CONFIG} seed=${SEED}"
done
