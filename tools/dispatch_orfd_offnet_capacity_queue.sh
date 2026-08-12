#!/usr/bin/env bash
set -euo pipefail

# Capacity-aware continuation for the remaining OFF-Net ORFD seeds.  This
# script only controls admission to the GPU; the formal OFF-Net recipe is
# kept verbatim (30 epochs, 704x1280, batch 2, accumulation 4, FP32).
# Seed 40 is reserved for the independent GPU0 watcher, so this queue owns
# seeds 41 and 42 on GPU1.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-1}"
ENV_NAME="${ENV_NAME:-litevilnet_roadformer_ral}"
ORFD_ROOT="${ORFD_ROOT:-$PWD/runs/revision_1/matched_orfd/local_data/Final_Dataset}"
NORMAL_ROOT="${NORMAL_ROOT:-$PWD/runs/revision_1/matched_orfd/local_exact_normals/offnet}"
SOURCE="${SOURCE:-$PWD/third_party/matched_baselines/OFF-Net}"
LOG="${LOG:-/tmp/litevilnet_orfd_offnet_capacity_queue.log}"
SEEDS="${SEEDS:-41 42}"
CAPACITY_LIMIT="${OFFNET_CAPACITY_LIMIT:-30000}"
LOCK_DIR="${LOCK_DIR:-${ROOT}/.offnet-capacity-queue.lock}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  log "another OFF-Net capacity queue owns ${LOCK_DIR}; exiting"
  exit 0
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

gpu_memory() {
  nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits \
    | tr -d '[:space:]' | head -n 1
}

wait_capacity() {
  local memory
  while :; do
    memory="$(gpu_memory 2>/dev/null || true)"
    if [[ "${memory:-999999}" =~ ^[0-9]+$ ]] && (( memory < CAPACITY_LIMIT )); then
      log "GPU${GPU} capacity available (${memory} MiB used; limit ${CAPACITY_LIMIT} MiB)"
      return 0
    fi
    log "GPU${GPU} busy (${memory:-unknown} MiB used); waiting"
    sleep 60
  done
}

wait_result() {
  local seed="$1"
  while [[ ! -s "${ROOT}/offnet_seed${seed}/result.json" ]]; do
    log "waiting for offnet_seed${seed}/result.json"
    sleep 60
  done
}

for seed in ${SEEDS}; do
  output_dir="${ROOT}/offnet_seed${seed}"
  if [[ -s "${output_dir}/result.json" ]]; then
    log "offnet_seed${seed} already completed; skipping"
    continue
  fi
  if [[ ! -s "${output_dir}/best_model.pth" ]]; then
    log "missing checkpoint: ${output_dir}/best_model.pth"
    exit 1
  fi
  if pgrep -af "[t]rain_matched_orfd_baseline.py.*--baseline offnet.*--seed ${seed}( |$)" >/dev/null; then
    log "offnet_seed${seed} already running elsewhere; waiting for its result"
    wait_result "${seed}"
    continue
  fi
  wait_capacity
  if [[ -s "${output_dir}/result.json" ]]; then
    log "offnet_seed${seed} completed while waiting; skipping"
    continue
  fi
  if pgrep -af "[t]rain_matched_orfd_baseline.py.*--baseline offnet.*--seed ${seed}( |$)" >/dev/null; then
    log "offnet_seed${seed} was claimed while waiting; waiting for its result"
    wait_result "${seed}"
    continue
  fi
  log "resuming OFF-Net seed-${seed} on GPU${GPU} from ${output_dir}/best_model.pth"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
    python tools/train_matched_orfd_baseline.py --baseline offnet \
      --official-source "${SOURCE}" --data-root "${ORFD_ROOT}" \
      --normal-root "${NORMAL_ROOT}" --output-dir "${output_dir}" \
      --seed "${seed}" --epochs 30 --resume "${output_dir}/best_model.pth" \
      --batch-size 2 --num-workers 8 --gradient-accumulation-steps 4 \
      --height 704 --width 1280 --val-every 1 --device cuda
  log "OFF-Net seed-${seed} process exited"
done
log "capacity-aware OFF-Net continuation completed"
