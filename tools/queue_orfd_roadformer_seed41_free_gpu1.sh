#!/usr/bin/env bash
set -euo pipefail

# Start RoadFormer seed-41 on GPU1 as soon as that card is genuinely free.
# This is scheduling-only: the formal recipe remains 50 epochs, batch 4,
# FP32, and 704x1280 input.  The existing seed-40/41/42 queue remains the
# fallback and will skip this seed if it has already completed.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-1}"
SEED="${SEED:-41}"
ENV_NAME="${ENV_NAME:-litevilnet_roadformer_ral}"
ROADFORMER_ROOT="${ROADFORMER_ROOT:-$PWD/runs/revision_1/matched_orfd/roadformer_orfd_exact}"
ROADFORMER_SOURCE="${ROADFORMER_SOURCE:-$PWD/third_party/matched_baselines/Road-Former}"
LOG="${LOG:-/tmp/litevilnet_orfd_roadformer_seed41_free_gpu1.log}"
LOCK_DIR="${LOCK_DIR:-${ROOT}/.roadformer-seed${SEED}-free-gpu1.lock}"
# RoadFormer can approach the full 48-GB card during attention.  Leave only
# the desktop/Xorg footprint before admission; this avoids co-residency with
# another training job and does not change any training hyperparameter.
CAPACITY_LIMIT="${ROADFORMER_FREE_GPU1_LIMIT:-5000}"
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:32}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  log "another RoadFormer seed-41 GPU1 watcher owns ${LOCK_DIR}; exiting"
  exit 0
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

OUTPUT_DIR="${ROOT}/roadformer_seed${SEED}"
mkdir -p "${OUTPUT_DIR}"

gpu_memory() {
  nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits \
    2>/dev/null | tr -d '[:space:]' | head -n 1
}

seed_running() {
  pgrep -af "[t]rain_matched_orfd_roadformer.py.*--seed ${SEED}( |$)" >/dev/null
}

while [[ ! -s "${OUTPUT_DIR}/result.json" ]]; do
  if seed_running; then
    log "RoadFormer seed-${SEED} is already running; waiting for result"
    sleep 60
    continue
  fi

  memory="$(gpu_memory || true)"
  if ! [[ "${memory:-999999}" =~ ^[0-9]+$ ]] || (( memory >= CAPACITY_LIMIT )); then
    log "GPU${GPU} not empty (${memory:-unknown} MiB; limit ${CAPACITY_LIMIT}); waiting"
    sleep 60
    continue
  fi

  log "starting RoadFormer seed-${SEED} on free GPU${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}" \
    PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n "${ENV_NAME}" \
    env PYTHONPATH=. python tools/train_matched_orfd_roadformer.py \
      --official-source "${ROADFORMER_SOURCE}" \
      --data-root "${ROADFORMER_ROOT}" --output-dir "${OUTPUT_DIR}" \
      --seed "${SEED}" --epochs 50 --batch-size 4 --num-workers 8 \
      --height 704 --width 1280 --val-every 1
  log "RoadFormer seed-${SEED} process exited"
  [[ -s "${OUTPUT_DIR}/result.json" ]] || sleep 30
done

log "RoadFormer seed-${SEED} result detected; watcher exiting"
