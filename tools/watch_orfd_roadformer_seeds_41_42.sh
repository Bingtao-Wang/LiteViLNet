#!/usr/bin/env bash
set -euo pipefail

# Resilient continuation for RoadFormer seeds 41 and 42.  This watchdog only
# restarts an absent process from the latest epoch-boundary checkpoint; the
# formal training recipe is kept verbatim.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU_BY_SEED_41="${GPU_BY_SEED_41:-1}"
GPU_BY_SEED_42="${GPU_BY_SEED_42:-0}"
ENV_NAME="${ENV_NAME:-litevilnet_roadformer_ral}"
ROADFORMER_ROOT="${ROADFORMER_ROOT:-$PWD/runs/revision_1/matched_orfd/roadformer_orfd_exact}"
ROADFORMER_SOURCE="${ROADFORMER_SOURCE:-$PWD/third_party/matched_baselines/Road-Former}"
LOG="${LOG:-/tmp/litevilnet_orfd_roadformer_41_42_watchdog.log}"
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:32}"
LOCK_DIR="${LOCK_DIR:-${ROOT}/.roadformer-41-42-watchdog.lock}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  log "another RoadFormer 41/42 watchdog owns ${LOCK_DIR}; exiting"
  exit 0
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

gpu_for_seed() {
  case "$1" in
    41) printf '%s\n' "${GPU_BY_SEED_41}" ;;
    42) printf '%s\n' "${GPU_BY_SEED_42}" ;;
    *) return 1 ;;
  esac
}

seed_running() {
  local seed="$1"
  pgrep -af "[t]rain_matched_orfd_roadformer.py.*--seed ${seed}( |$)" >/dev/null
}

wait_capacity() {
  local gpu="$1" memory
  while :; do
    memory="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits \
      2>/dev/null | tr -d '[:space:]' | head -n 1 || true)"
    if [[ "${memory:-999999}" =~ ^[0-9]+$ ]] && (( memory < 30000 )); then
      log "GPU${gpu} capacity available (${memory} MiB used; limit 30000 MiB)"
      return 0
    fi
    log "GPU${gpu} busy (${memory:-unknown} MiB used); waiting"
    sleep 60
  done
}

run_seed() {
  local seed="$1" gpu output_dir checkpoint
  local -a resume_args=()
  gpu="$(gpu_for_seed "${seed}")"
  output_dir="${ROOT}/roadformer_seed${seed}"
  mkdir -p "${output_dir}"
  while [[ ! -s "${output_dir}/result.json" ]]; do
    if seed_running "${seed}"; then
      log "roadformer_seed${seed} is running; waiting"
      sleep 60
      continue
    fi
    checkpoint="${output_dir}/best_model.pth"
    resume_args=()
    if [[ -s "${checkpoint}" ]]; then
      resume_args=(--resume "${checkpoint}")
    fi
    wait_capacity "${gpu}"
    if seed_running "${seed}" || [[ -s "${output_dir}/result.json" ]]; then
      continue
    fi
    log "launching RoadFormer seed-${seed} on GPU${gpu}${resume_args[*]:+ from checkpoint}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}" \
      PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n "${ENV_NAME}" \
      env PYTHONPATH=. python tools/train_matched_orfd_roadformer.py \
        --official-source "${ROADFORMER_SOURCE}" \
        --data-root "${ROADFORMER_ROOT}" --output-dir "${output_dir}" \
        --seed "${seed}" --epochs 50 "${resume_args[@]}" \
        --batch-size 4 --num-workers 8 --height 704 --width 1280 --val-every 1
    log "RoadFormer seed-${seed} process exited; checking result"
  done
  log "roadformer_seed${seed}/result.json detected"
}

for seed in 41 42; do
  run_seed "${seed}" &
done
wait
log 'RoadFormer seed 41/42 watchdog completed'
