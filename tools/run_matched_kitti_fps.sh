#!/usr/bin/env bash
set -euo pipefail

# Run all five Table-I FPS-1 measurements sequentially on one otherwise idle
# RTX 4090 D. Activate litevilnet_ral before invoking this script; RoadFormer
# is launched through its separately pinned environment.

: "${LITEVILNET_CHECKPOINT:?Set LITEVILNET_CHECKPOINT to a full-model best checkpoint}"
: "${USNET_CHECKPOINT:?Set USNET_CHECKPOINT to a matched USNet best checkpoint}"
: "${SNE_CHECKPOINT:?Set SNE_CHECKPOINT to a matched SNE-RoadSeg best checkpoint}"
: "${PLARD_CHECKPOINT:?Set PLARD_CHECKPOINT to a matched PLARD best checkpoint}"
: "${ROADFORMER_CHECKPOINT:?Set ROADFORMER_CHECKPOINT to a matched RoadFormer best checkpoint}"
: "${USNET_SOURCE:?Set USNET_SOURCE to the pinned official USNet clone}"
: "${SNE_SOURCE:?Set SNE_SOURCE to the pinned official SNE-RoadSeg clone}"
: "${PLARD_SOURCE:?Set PLARD_SOURCE to the pinned official PLARD clone}"
: "${ROADFORMER_SOURCE:?Set ROADFORMER_SOURCE to the pinned official RoadFormer clone}"

FPS_GPU="${FPS_GPU:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/revision_1/matched_baselines/fps_4090d}"
COMMON_ARGS=(--precision fp32 --warmup 100 --iterations 300 --repeats 3)
mkdir -p "${OUTPUT_ROOT}"

run_standard() {
  local method="$1"
  local checkpoint="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="${FPS_GPU}" python tools/benchmark_matched_kitti_fps.py \
    --method "${method}" \
    --checkpoint "${checkpoint}" \
    "${COMMON_ARGS[@]}" \
    --output "${OUTPUT_ROOT}/${method}.json" \
    "$@"
}

run_standard litevilnet "${LITEVILNET_CHECKPOINT}"
run_standard usnet "${USNET_CHECKPOINT}" --official-source "${USNET_SOURCE}"
run_standard sne_roadseg "${SNE_CHECKPOINT}" --official-source "${SNE_SOURCE}"
run_standard plard "${PLARD_CHECKPOINT}" --official-source "${PLARD_SOURCE}"

CUDA_VISIBLE_DEVICES="${FPS_GPU}" conda run --no-capture-output \
  -n "${ROADFORMER_ENV:-litevilnet_roadformer_ral}" env PYTHONPATH=. \
  python tools/benchmark_matched_kitti_fps.py \
  --method roadformer \
  --official-source "${ROADFORMER_SOURCE}" \
  --checkpoint "${ROADFORMER_CHECKPOINT}" \
  "${COMMON_ARGS[@]}" \
  --output "${OUTPUT_ROOT}/roadformer.json"

python tools/summarize_matched_kitti_fps.py \
  --input-root "${OUTPUT_ROOT}" \
  --output-json docs/ral/table1_matched_baselines/results/fps_4090d_summary.json \
  --output-csv docs/ral/table1_matched_baselines/results/fps_4090d_summary.csv \
  --result-output-dir docs/ral/table1_matched_baselines/results/fps \
  --anonymous-result-copies
