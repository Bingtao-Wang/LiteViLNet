#!/usr/bin/env bash
set -euo pipefail

: "${OFFNET_SOURCE:?Set OFFNET_SOURCE to the pinned official OFF-Net clone}"
: "${MATCHED_ROOT:?Set MATCHED_ROOT to the prepared matched KITTI tree}"
: "${KITTI_OFFNET_NORMAL_ROOT:?Set KITTI_OFFNET_NORMAL_ROOT to the OFF-Net SNE cache}"

OUTPUT_ROOT="${OUTPUT_ROOT:-runs/revision_1/matched_baselines/formal}"
SEEDS="${SEEDS:-40 41 42}"
for seed in ${SEEDS}; do
  if [[ -s "${OUTPUT_ROOT}/offnet_seed${seed}/result.json" ]]; then
    echo "Skipping completed KITTI OFF-Net seed ${seed}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${OFFNET_GPU:-0}" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "${OPENMMLAB_ENV:-litevilnet_roadformer_ral}" \
    env PYTHONPATH=. python tools/train_matched_kitti_offnet.py \
    --official-source "${OFFNET_SOURCE}" --data-root "${MATCHED_ROOT}" \
    --normal-root "${KITTI_OFFNET_NORMAL_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/offnet_seed${seed}" \
    --seed "${seed}" --epochs 150 --batch-size 2 --num-workers 4 \
    --height 384 --width 1248 --val-every 5 --early-stop-validations 20
done
