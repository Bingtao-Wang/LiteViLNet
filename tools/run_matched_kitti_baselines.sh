#!/usr/bin/env bash
set -euo pipefail

# Required environment variables point to clean clones at the commits listed
# in docs/ral/table1_matched_baselines/README.md.
: "${USNET_SOURCE:?Set USNET_SOURCE to the official USNet clone}"
: "${SNE_SOURCE:?Set SNE_SOURCE to the official SNE-RoadSeg clone}"
: "${PLARD_SOURCE:?Set PLARD_SOURCE to the official PLARD clone}"
: "${ROADFORMER_SOURCE:?Set ROADFORMER_SOURCE to the official RoadFormer clone}"
: "${MATCHED_ROOT:?Set MATCHED_ROOT to the prepared matched KITTI tree}"
: "${ROADFORMER_DATA_ROOT:?Set ROADFORMER_DATA_ROOT to the prepared RoadFormer KITTI tree}"

OUTPUT_ROOT="${OUTPUT_ROOT:-runs/revision_1/matched_baselines/formal}"
SEEDS="${SEEDS:-40 41 42}"
EPOCHS="${EPOCHS:-150}"

run_usnet() {
  for seed in ${SEEDS}; do
    CUDA_VISIBLE_DEVICES="${USNET_GPU:-0}" python tools/train_matched_kitti_baseline.py \
      --baseline usnet \
      --official-source "${USNET_SOURCE}" \
      --data-root "${MATCHED_ROOT}" \
      --output-dir "${OUTPUT_ROOT}/usnet_seed${seed}" \
      --seed "${seed}" --epochs "${EPOCHS}" --batch-size 2 --num-workers 4 \
      --val-every 5 --early-stop-validations 20 --amp --device cuda
  done
}

run_sne_roadseg() {
  for seed in ${SEEDS}; do
    CUDA_VISIBLE_DEVICES="${SNE_GPU:-1}" python tools/train_matched_kitti_baseline.py \
      --baseline sne_roadseg \
      --official-source "${SNE_SOURCE}" \
      --data-root "${MATCHED_ROOT}" \
      --output-dir "${OUTPUT_ROOT}/sne_roadseg_seed${seed}" \
      --seed "${seed}" --epochs "${EPOCHS}" --batch-size 2 --num-workers 4 \
      --val-every 5 --early-stop-validations 20 --amp --device cuda
  done
}

run_plard() {
  for seed in ${SEEDS}; do
    CUDA_VISIBLE_DEVICES="${USNET_GPU:-0}" python tools/train_matched_kitti_baseline.py \
      --baseline plard \
      --official-source "${PLARD_SOURCE}" \
      --data-root "${MATCHED_ROOT}" \
      --output-dir "${OUTPUT_ROOT}/plard_seed${seed}" \
      --seed "${seed}" --epochs "${EPOCHS}" --batch-size 4 --num-workers 4 \
      --val-every 5 --early-stop-validations 20 --device cuda
  done
}

run_roadformer() {
  for seed in ${SEEDS}; do
    CUDA_VISIBLE_DEVICES="${SNE_GPU:-1}" conda run --no-capture-output \
      -n "${ROADFORMER_ENV:-litevilnet_roadformer_ral}" env PYTHONPATH=. \
      python tools/train_matched_kitti_roadformer.py \
      --official-source "${ROADFORMER_SOURCE}" \
      --data-root "${ROADFORMER_DATA_ROOT}" \
      --output-dir "${OUTPUT_ROOT}/roadformer_seed${seed}" \
      --seed "${seed}" --epochs "${EPOCHS}" --batch-size 4 --num-workers 4 \
      --val-every 5 --early-stop-validations 20
  done
}

run_gpu0_queue() {
  run_usnet
  run_plard
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

python tools/summarize_matched_kitti_baselines.py \
  --input-root "${OUTPUT_ROOT}" \
  --expected-seeds "$(printf '%s' "${SEEDS}" | tr ' ' ',')" \
  --output-json docs/ral/table1_matched_baselines/results/summary.json \
  --output-csv docs/ral/table1_matched_baselines/results/summary.csv \
  --seed-output-dir docs/ral/table1_matched_baselines/results/seeds \
  --anonymous-seed-copies
