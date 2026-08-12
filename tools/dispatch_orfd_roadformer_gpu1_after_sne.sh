#!/usr/bin/env bash
set -euo pipefail

# Capacity-aware second RoadFormer lane.  GPU0 owns the original lane; this
# helper waits until the GPU1 SNE queue has released its slot, then takes the
# first missing RoadFormer seed(s).  It never changes the official recipe and
# waits for an already-running seed instead of launching a duplicate.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-1}"
ENV_NAME="${ENV_NAME:-litevilnet_roadformer_ral}"
ROADFORMER_ROOT="${ROADFORMER_ROOT:-$PWD/runs/revision_1/matched_orfd/roadformer_orfd_exact}"
ROADFORMER_SOURCE="${ROADFORMER_SOURCE:-$PWD/third_party/matched_baselines/Road-Former}"
LOG="${LOG:-/tmp/litevilnet_orfd_roadformer_gpu1_after_sne.log}"
GPU_MEMORY_LIMIT="${GPU_MEMORY_LIMIT:-30000}"
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:32}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

wait_result() {
  local name="$1"
  while [[ ! -s "${ROOT}/${name}/result.json" ]]; do
    log "waiting for ${name}/result.json"
    sleep 60
  done
}

wait_gpu_capacity() {
  local memory
  while :; do
    memory="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
      | tr -d '[:space:]' | head -n 1 || true)"
    if [[ "${memory:-999999}" =~ ^[0-9]+$ ]] && (( memory < GPU_MEMORY_LIMIT )); then
      log "GPU${GPU} capacity available (${memory} MiB used; limit ${GPU_MEMORY_LIMIT} MiB)"
      return
    fi
    log "GPU${GPU} busy (${memory:-unknown} MiB used); waiting"
    sleep 60
  done
}

wait_running_or_result() {
  local seed="$1" name="roadformer_seed${1}"
  while :; do
    [[ -s "${ROOT}/${name}/result.json" ]] && return 0
    if pgrep -af "[t]rain_matched_orfd_roadformer.py.*roadformer_seed${seed}" >/dev/null; then
      log "${name} is already running; waiting for its result"
      sleep 60
    else
      return 1
    fi
  done
}

# SNE-RoadSeg is the last GPU1 job in the first lane.  This dependency keeps
# the two high-memory networks from colliding on the 48-GB card.
wait_result sne_roadseg_seed42

for seed in 41 42; do
  name="roadformer_seed${seed}"
  if [[ -s "${ROOT}/${name}/result.json" ]]; then
    log "${name} already completed; skipping"
    continue
  fi
  if wait_running_or_result "${seed}"; then
    log "${name} completed while waiting; continuing"
    continue
  fi
  wait_gpu_capacity
  if [[ -s "${ROOT}/${name}/result.json" ]]; then
    continue
  fi
  if wait_running_or_result "${seed}"; then
    log "${name} completed while waiting for GPU capacity; continuing"
    continue
  fi
  mkdir -p "${ROOT}/${name}"
  log "launching RoadFormer seed-${seed} on GPU${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
    python tools/train_matched_orfd_roadformer.py \
      --official-source "${ROADFORMER_SOURCE}" \
      --data-root "${ROADFORMER_ROOT}" --output-dir "${ROOT}/${name}" \
      --seed "${seed}" --epochs 50 --batch-size 4 --num-workers 8 \
      --height 704 --width 1280 --val-every 1
  log "${name} process exited"
done
log "GPU1 RoadFormer continuation completed"
