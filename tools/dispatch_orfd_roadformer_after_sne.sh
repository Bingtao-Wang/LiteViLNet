#!/usr/bin/env bash
set -euo pipefail

# Capacity-aware continuation for RoadFormer only.  The other ORFD jobs are
# already owned by the first-wave/follow-up dispatchers; this helper avoids
# claiming or restarting them.  It keeps the official RoadFormer recipe
# unchanged and starts its three seeds sequentially after the SNE jobs that
# occupy the high-memory GPU0 slot have finished.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-0}"
ENV_NAME="${ENV_NAME:-litevilnet_roadformer_ral}"
ROADFORMER_ROOT="${ROADFORMER_ROOT:-$PWD/runs/revision_1/matched_orfd/roadformer_orfd_exact}"
ROADFORMER_SOURCE="${ROADFORMER_SOURCE:-$PWD/third_party/matched_baselines/Road-Former}"
LOG="${LOG:-/tmp/litevilnet_orfd_roadformer_dispatcher.log}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }
wait_result() {
  local d="$1"
  while [[ ! -s "${ROOT}/${d}/result.json" ]]; do
    log "waiting for ${d}/result.json"
    sleep 60
  done
  log "found completed ${d}"
}
gpu_mem() {
  nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits \
    | tr -d '[:space:]' | head -n 1
}
wait_capacity() {
  local limit="${1:-30000}" memory
  while :; do
    memory="$(gpu_mem || true)"
    if [[ "${memory:-999999}" =~ ^[0-9]+$ ]] && (( memory < limit )); then
      log "GPU${GPU} capacity available (${memory} MiB used; limit ${limit} MiB)"
      return
    fi
    log "GPU${GPU} busy (${memory:-unknown} MiB used); waiting"
    sleep 60
  done
}

# SNE seed-41 is scheduled independently after USNet seed-42 and may share
# GPU0 with seed-40.  Waiting for it here preserves the previous high-memory
# isolation guarantee before RoadFormer's batch-4 training begins.
wait_result sne_roadseg_seed41
wait_capacity 30000

for seed in 40 41 42; do
  output_dir="${ROOT}/roadformer_seed${seed}"
  if [[ -s "${output_dir}/result.json" ]]; then
    log "skipping completed roadformer_seed${seed}"
    continue
  fi
  if pgrep -af "train_matched_orfd_roadformer.py.*roadformer_seed${seed}" >/dev/null; then
    log "roadformer_seed${seed} is already running; stopping continuation"
    exit 0
  fi
  mkdir -p "${output_dir}"
  log "launching RoadFormer seed-${seed} on GPU${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
    python tools/train_matched_orfd_roadformer.py \
      --official-source "${ROADFORMER_SOURCE}" \
      --data-root "${ROADFORMER_ROOT}" --output-dir "${output_dir}" \
      --seed "${seed}" --epochs 50 --batch-size 4 --num-workers 8 \
      --height 704 --width 1280 --val-every 1 \
      >>"/tmp/litevilnet_orfd_roadformer${seed}.log" 2>&1
  log "RoadFormer seed-${seed} process exited"
done
log "RoadFormer continuation completed"
