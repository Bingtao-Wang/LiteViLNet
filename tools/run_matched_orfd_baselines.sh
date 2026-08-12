#!/usr/bin/env bash
set -euo pipefail

: "${ORFD_ROOT:?Set ORFD_ROOT to the released Final_Dataset directory}"
: "${ORFD_SNE_NORMAL_ROOT:?Set ORFD_SNE_NORMAL_ROOT to the SNE-RoadSeg cache}"
: "${ORFD_OFFNET_NORMAL_ROOT:?Set ORFD_OFFNET_NORMAL_ROOT to the OFF-Net cache}"
: "${ORFD_ROADFORMER_ROOT:?Set ORFD_ROADFORMER_ROOT to the prepared RoadFormer tree}"
: "${USNET_SOURCE:?Set USNET_SOURCE to the pinned official USNet clone}"
: "${SNE_SOURCE:?Set SNE_SOURCE to the pinned official SNE-RoadSeg clone}"
: "${OFFNET_SOURCE:?Set OFFNET_SOURCE to the pinned official OFF-Net clone}"
: "${ROADFORMER_SOURCE:?Set ROADFORMER_SOURCE to the pinned official RoadFormer clone}"

OUTPUT_ROOT="${OUTPUT_ROOT:-runs/revision_1/matched_orfd/formal}"
SEEDS="${SEEDS:-40 41 42}"
ROADFORMER_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:32}"

run_usnet() {
  for seed in ${SEEDS}; do
    if [[ -s "${OUTPUT_ROOT}/usnet_seed${seed}/result.json" ]]; then
      echo "Skipping completed ORFD USNet seed ${seed}"
      continue
    fi
    CUDA_VISIBLE_DEVICES="${GPU0:-0}" PYTHONDONTWRITEBYTECODE=1 \
      conda run --no-capture-output -n "${MAIN_ENV:-litevilnet_ral}" env PYTHONPATH=. \
      python tools/train_matched_orfd_baseline.py \
      --baseline usnet --official-source "${USNET_SOURCE}" \
      --data-root "${ORFD_ROOT}" --normal-root "${ORFD_SNE_NORMAL_ROOT}" \
      --output-dir "${OUTPUT_ROOT}/usnet_seed${seed}" \
      --seed "${seed}" --epochs 30 --batch-size 2 --num-workers 8 \
      --height 704 --width 1280 --val-every 1 --amp --device cuda
  done
}

run_sne_roadseg() {
  for seed in ${SEEDS}; do
    if [[ -s "${OUTPUT_ROOT}/sne_roadseg_seed${seed}/result.json" ]]; then
      echo "Skipping completed ORFD SNE-RoadSeg seed ${seed}"
      continue
    fi
    CUDA_VISIBLE_DEVICES="${GPU1:-1}" PYTHONDONTWRITEBYTECODE=1 \
      conda run --no-capture-output -n "${MAIN_ENV:-litevilnet_ral}" env PYTHONPATH=. \
      python tools/train_matched_orfd_baseline.py \
      --baseline sne_roadseg --official-source "${SNE_SOURCE}" \
      --data-root "${ORFD_ROOT}" --normal-root "${ORFD_SNE_NORMAL_ROOT}" \
      --output-dir "${OUTPUT_ROOT}/sne_roadseg_seed${seed}" \
      --seed "${seed}" --epochs 30 --batch-size 2 --num-workers 8 \
      --height 704 --width 1280 --val-every 1 --amp --device cuda
  done
}

run_offnet() {
  for seed in ${SEEDS}; do
    if [[ -s "${OUTPUT_ROOT}/offnet_seed${seed}/result.json" ]]; then
      echo "Skipping completed ORFD OFF-Net seed ${seed}"
      continue
    fi
    CUDA_VISIBLE_DEVICES="${GPU0:-0}" PYTHONDONTWRITEBYTECODE=1 \
      conda run --no-capture-output -n "${OPENMMLAB_ENV:-litevilnet_roadformer_ral}" env PYTHONPATH=. \
      python tools/train_matched_orfd_baseline.py \
      --baseline offnet --official-source "${OFFNET_SOURCE}" \
      --data-root "${ORFD_ROOT}" --normal-root "${ORFD_OFFNET_NORMAL_ROOT}" \
      --output-dir "${OUTPUT_ROOT}/offnet_seed${seed}" \
      --seed "${seed}" --epochs 30 --batch-size 2 --num-workers 8 \
      --gradient-accumulation-steps 4 \
      --height 704 --width 1280 --val-every 1 --device cuda
  done
}

run_roadformer() {
  for seed in ${SEEDS}; do
    if [[ -s "${OUTPUT_ROOT}/roadformer_seed${seed}/result.json" ]]; then
      echo "Skipping completed ORFD RoadFormer seed ${seed}"
      continue
    fi
    CUDA_VISIBLE_DEVICES="${GPU1:-1}" PYTORCH_CUDA_ALLOC_CONF="${ROADFORMER_ALLOC_CONF}" PYTHONDONTWRITEBYTECODE=1 \
      conda run --no-capture-output -n "${OPENMMLAB_ENV:-litevilnet_roadformer_ral}" env PYTHONPATH=. \
      python tools/train_matched_orfd_roadformer.py \
      --official-source "${ROADFORMER_SOURCE}" --data-root "${ORFD_ROADFORMER_ROOT}" \
      --output-dir "${OUTPUT_ROOT}/roadformer_seed${seed}" \
      --seed "${seed}" --epochs 50 --batch-size 4 --num-workers 8 \
      --height 704 --width 1280 --val-every 1
  done
}

run_gpu0_queue() {
  run_usnet
  run_offnet
}

run_gpu1_queue() {
  run_sne_roadseg
  run_roadformer
}

run_gpu0_queue &
gpu0_pid=$!
run_gpu1_queue &
gpu1_pid=$!
wait "${gpu0_pid}"
wait "${gpu1_pid}"

python tools/summarize_matched_orfd_baselines.py \
  --input-root "${OUTPUT_ROOT}" \
  --expected-seeds "$(printf '%s' "${SEEDS}" | tr ' ' ',')" \
  --output-json docs/ral/orfd_matched_baselines/results/summary.json \
  --output-csv docs/ral/orfd_matched_baselines/results/summary.csv \
  --seed-output-dir docs/ral/orfd_matched_baselines/results/seeds
