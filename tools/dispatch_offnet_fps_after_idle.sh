#!/usr/bin/env bash
set -euo pipefail

# Measure the locally retrained OFF-Net KITTI FPS-1 only after the selected
# RTX 4090 D is genuinely idle.  This avoids reporting a timing contaminated
# by another training or external compute job and leaves all such jobs alone.

GPU="${FPS_GPU:-0}"
CHECKPOINT="${OFFNET_CHECKPOINT:-runs/revision_1/matched_baselines/formal/offnet_seed42/best_model.pth}"
SOURCE="${OFFNET_SOURCE:-$PWD/third_party/matched_baselines/OFF-Net}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/revision_1/matched_baselines/fps_4090d}"
LOG="${LOG:-/tmp/litevilnet_offnet_fps_idle.log}"

mkdir -p "${OUTPUT_ROOT}"
exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

while :; do
  if [[ ! -s "${CHECKPOINT}" ]]; then
    log "OFF-Net checkpoint is not ready (${CHECKPOINT}); waiting"
    sleep 60
    continue
  fi
  active="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
  if [[ -z "${active}" ]]; then
    break
  fi
  log "GPU${GPU} still has compute PIDs (${active//$'\n'/,}); waiting"
  sleep 60
done

log "GPU${GPU} is idle; measuring OFF-Net FPS-1"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONDONTWRITEBYTECODE=1 \
  conda run --no-capture-output -n "${ROADFORMER_ENV:-litevilnet_roadformer_ral}" \
  env PYTHONPATH=. python tools/benchmark_matched_kitti_fps.py \
  --method offnet --official-source "${SOURCE}" --checkpoint "${CHECKPOINT}" \
  --precision fp32 --warmup 100 --iterations 300 --repeats 3 \
  --output "${OUTPUT_ROOT}/offnet.json"

conda run --no-capture-output -n "${MAIN_ENV:-litevilnet_ral}" \
  env PYTHONPATH=. python tools/summarize_matched_kitti_fps.py \
  --input-root "${OUTPUT_ROOT}" \
  --output-json docs/ral/table1_matched_baselines/results/fps_4090d_summary.json \
  --output-csv docs/ral/table1_matched_baselines/results/fps_4090d_summary.csv \
  --result-output-dir docs/ral/table1_matched_baselines/results/fps \
  --anonymous-result-copies
log "OFF-Net FPS-1 evidence and aggregate summary completed"
