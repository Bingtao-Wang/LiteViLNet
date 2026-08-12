#!/usr/bin/env bash
set -euo pipefail

# Capacity-aware multi-GPU SNE-RoadSeg continuation.  SNE-RoadSeg peaks at
# roughly 34.7 GiB on the formal ORFD recipe, so at most one SNE worker is
# admitted per GPU and only below a conservative occupancy threshold.  This
# scheduler owns launch ordering only; all method
# hyperparameters and evaluation settings remain unchanged.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
# Keep GPU0 reserved for the high-memory RoadFormer lane.  Operators can
# still override GPU_LIST explicitly when scheduling on a different host.
GPU_LIST="${GPU_LIST:-1}"
ENV_NAME="${ENV_NAME:-litevilnet_ral}"
# Scheduling guard only; the default remains the original conservative limit.
# Raising it is safe only after checking the companion job's measured peak and
# does not change any SNE training hyperparameter.
# Allow a SNE worker to join a single OFF-Net worker (about 13 GB at
# admission) while keeping the card reserved when two heavy workers are
# already resident.  This is a scheduling guard only; it does not alter the
# SNE training recipe.
CAPACITY_LIMIT="${SNE_CAPACITY_LIMIT:-20000}"
ORFD_ROOT="${ORFD_ROOT:-$PWD/runs/revision_1/matched_orfd/local_data/Final_Dataset}"
NORMAL_ROOT="${NORMAL_ROOT:-$PWD/runs/revision_1/matched_orfd/local_exact_normals/sne_roadseg}"
SOURCE="${SOURCE:-$PWD/third_party/matched_baselines/SNE-RoadSeg}"
LOG="${LOG:-/tmp/litevilnet_orfd_sne_capacity_queue.log}"
LOCK_DIR="${LOCK_DIR:-${ROOT}/.sne-capacity-queue.lock}"
GPU_LOCK_ROOT="${GPU_LOCK_ROOT:-${ROOT}/.sne-capacity-gpu-locks}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  log "another SNE capacity queue owns ${LOCK_DIR}; exiting"
  exit 0
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

wait_result() {
  local name="$1"
  while [[ ! -s "${ROOT}/${name}/result.json" ]]; do
    log "waiting for ${name}/result.json"
    sleep 60
  done
  log "found completed ${name}"
}

wait_capacity() {
  local gpu="$1" memory
  while :; do
    memory="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
      | tr -d '[:space:]' | head -n 1 || true)"
    if [[ "${memory:-999999}" =~ ^[0-9]+$ ]] && (( memory < CAPACITY_LIMIT )); then
      log "GPU${gpu} capacity available (${memory} MiB used; limit ${CAPACITY_LIMIT} MiB)"
      return 0
    fi
    log "GPU${gpu} busy (${memory:-unknown} MiB used); waiting"
    sleep 60
  done
}

# SNE-RoadSeg is an independent baseline and does not consume any OFF-Net or
# USNet checkpoint.  Earlier versions waited for the entire OFF-Net lane here,
# which could unnecessarily delay SNE when the remaining OFF-Net seed was on
# the other GPU.  Keep an opt-in dependency gate for operators who want the
# old ordering, but by default launch as soon as the memory guard is safe.
if [[ "${WAIT_FOR_DEPENDENCIES:-0}" == "1" ]]; then
  for dependency in offnet_seed40 offnet_seed41 offnet_seed42 usnet_seed42; do
    wait_result "${dependency}"
  done
else
  log "SNE dependencies are independent; waiting only for GPU capacity"
fi

claim_gpu() {
  local gpu
  mkdir -p "${GPU_LOCK_ROOT}"
  for gpu in ${GPU_LIST}; do
    if mkdir "${GPU_LOCK_ROOT}/gpu${gpu}" 2>/dev/null; then
      printf '%s\n' "${gpu}"
      return 0
    fi
  done
  return 1
}

release_gpu() {
  local gpu="$1"
  rmdir "${GPU_LOCK_ROOT}/gpu${gpu}" 2>/dev/null || true
}

run_seed() {
  local seed="$1" output_dir gpu resume_args=()
  output_dir="${ROOT}/sne_roadseg_seed${seed}"
  if [[ -s "${output_dir}/result.json" ]]; then
    log "sne_roadseg_seed${seed} already completed; skipping"
    return 0
  fi
  if pgrep -af "[t]rain_matched_orfd_baseline.py.*--baseline sne_roadseg.*--seed ${seed}( |$)" >/dev/null; then
    log "sne_roadseg_seed${seed} already running elsewhere; waiting for its result"
    wait_result "sne_roadseg_seed${seed}"
    return 0
  fi
  mkdir -p "${output_dir}"
  if [[ -s "${output_dir}/best_model.pth" ]]; then
    resume_args=(--resume "${output_dir}/best_model.pth")
  fi

  while :; do
    if [[ -s "${output_dir}/result.json" ]]; then
      log "sne_roadseg_seed${seed} completed while waiting; skipping"
      return 0
    fi
    if pgrep -af "[t]rain_matched_orfd_baseline.py.*--baseline sne_roadseg.*--seed ${seed}( |$)" >/dev/null; then
      log "sne_roadseg_seed${seed} was claimed while waiting; waiting for its result"
      wait_result "sne_roadseg_seed${seed}"
      return 0
    fi
    gpu="$(claim_gpu || true)"
    if [[ -z "${gpu}" ]]; then
      log "all SNE GPU slots are occupied; seed-${seed} waiting"
      sleep 60
      continue
    fi
    # Reserve a slot before sampling memory so two workers cannot claim the
    # same card.  If another job still occupies the card, release the slot
    # and retry later rather than changing the formal recipe.
    memory="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
      | tr -d '[:space:]' | head -n 1 || true)"
    if ! [[ "${memory:-999999}" =~ ^[0-9]+$ ]] || (( memory >= CAPACITY_LIMIT )); then
      log "GPU${gpu} busy (${memory:-unknown} MiB used; limit ${CAPACITY_LIMIT} MiB); releasing slot for seed-${seed}"
      release_gpu "${gpu}"
      sleep 60
      continue
    fi
    log "launching SNE-RoadSeg seed-${seed} on GPU${gpu}${resume_args[*]:+ from checkpoint}"
    if CUDA_VISIBLE_DEVICES="${gpu}" PYTHONDONTWRITEBYTECODE=1 \
      conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
      python tools/train_matched_orfd_baseline.py --baseline sne_roadseg \
        --official-source "${SOURCE}" --data-root "${ORFD_ROOT}" \
        --normal-root "${NORMAL_ROOT}" --output-dir "${output_dir}" \
        --seed "${seed}" --epochs 30 "${resume_args[@]}" \
        --batch-size 2 --num-workers 8 --height 704 --width 1280 \
        --val-every 1 --amp --device cuda \
        >>"/tmp/litevilnet_orfd_sne_seed${seed}_queue.log" 2>&1; then
      log "SNE-RoadSeg seed-${seed} process exited"
    else
      log "SNE-RoadSeg seed-${seed} exited non-zero; retrying from checkpoint"
    fi
    release_gpu "${gpu}"
    if [[ -s "${output_dir}/result.json" ]]; then
      return 0
    fi
    sleep 30
  done
}

# Seeds are independent; launch up to one worker per GPU.  A shared result
# check prevents duplicate work when an older continuation already owns one.
for seed in 40 41 42; do
  run_seed "${seed}" &
done
wait
log "all SNE-RoadSeg seed workers completed"
