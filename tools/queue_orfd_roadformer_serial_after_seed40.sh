#!/usr/bin/env bash
set -euo pipefail

# Serial RoadFormer continuation for the 48-GB rebuttal host.  RoadFormer is
# the largest ORFD baseline; keeping one instance at a time avoids the host
# swap/allocator contention observed when seeds 40--42 were launched together.
# This file changes scheduling only.  Every training invocation below keeps
# the formal recipe (50 epochs, batch 4, 704x1280, 8 workers, FP32) unchanged.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-0}"
ENV_NAME="${ENV_NAME:-litevilnet_roadformer_ral}"
ROADFORMER_ROOT="${ROADFORMER_ROOT:-$PWD/runs/revision_1/matched_orfd/roadformer_orfd_exact}"
ROADFORMER_SOURCE="${ROADFORMER_SOURCE:-$PWD/third_party/matched_baselines/Road-Former}"
LOG="${LOG:-/tmp/litevilnet_orfd_roadformer_serial_after_seed40.log}"
CAPACITY_LIMIT="${ROADFORMER_CAPACITY_LIMIT:-30000}"
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:32}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

wait_result() {
  local name="$1"
  while [[ ! -s "${ROOT}/${name}/result.json" ]]; do
    log "waiting for ${name}/result.json"
    sleep 60
  done
  log "found ${name}/result.json"
}

wait_capacity() {
  local memory
  while :; do
    memory="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' | head -1 || true)"
    if [[ "${memory:-999999}" =~ ^[0-9]+$ ]] && (( memory < CAPACITY_LIMIT )); then
      log "GPU${GPU} capacity available (${memory} MiB used; limit ${CAPACITY_LIMIT} MiB)"
      return 0
    fi
    log "GPU${GPU} busy (${memory:-unknown} MiB used); waiting"
    sleep 60
  done
}

run_seed() {
  local seed="$1" output_dir checkpoint
  output_dir="${ROOT}/roadformer_seed${seed}"
  mkdir -p "${output_dir}"
  if [[ -s "${output_dir}/result.json" ]]; then
    log "roadformer_seed${seed} already completed; skipping"
    return 0
  fi
  while pgrep -af "[t]rain_matched_orfd_roadformer.py.*--seed ${seed}( |$)" >/dev/null; do
    log "roadformer_seed${seed} is already running; waiting"
    sleep 60
  done
  wait_capacity
  checkpoint="${output_dir}/best_model.pth"
  local -a resume_args=()
  if [[ -s "${checkpoint}" ]]; then
    resume_args=(--resume "${checkpoint}")
    log "resuming roadformer_seed${seed} from ${checkpoint}"
  else
    log "starting roadformer_seed${seed} from scratch"
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}" \
    PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n "${ENV_NAME}" \
    env PYTHONPATH=. python tools/train_matched_orfd_roadformer.py \
      --official-source "${ROADFORMER_SOURCE}" \
      --data-root "${ROADFORMER_ROOT}" --output-dir "${output_dir}" \
      --seed "${seed}" --epochs 50 "${resume_args[@]}" \
      --batch-size 4 --num-workers 8 --height 704 --width 1280 --val-every 1
  log "roadformer_seed${seed} process exited"
  [[ -s "${output_dir}/result.json" ]] || {
    log "roadformer_seed${seed} exited without result.json; retrying from checkpoint"
    sleep 30
    run_seed "${seed}"
  }
}

wait_result roadformer_seed40
for seed in 41 42; do
  run_seed "${seed}"
done
log 'serial RoadFormer seeds 41/42 completed'
