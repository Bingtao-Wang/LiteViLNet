#!/usr/bin/env bash
set -euo pipefail

# Conservative recovery dispatcher for OFF-Net seed 42 after an admission
# attempt hit CUDA OOM.  The epoch-6 checkpoint is valid; this helper only
# changes GPU admission/retry scheduling.  The formal recipe is unchanged:
# 30 epochs, batch 2, accumulation 4, FP32, and 704x1280 input.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU_LIST="${GPU_LIST:-0 1}"
ENV_NAME="${ENV_NAME:-litevilnet_roadformer_ral}"
ORFD_ROOT="${ORFD_ROOT:-$PWD/runs/revision_1/matched_orfd/local_data/Final_Dataset}"
NORMAL_ROOT="${NORMAL_ROOT:-$PWD/runs/revision_1/matched_orfd/local_exact_normals/offnet}"
SOURCE="${SOURCE:-$PWD/third_party/matched_baselines/OFF-Net}"
LOG="${LOG:-/tmp/litevilnet_orfd_offnet42_capacity_recovery.log}"
GPU_LOCK_ROOT="${GPU_LOCK_ROOT:-${ROOT}/.offnet42-capacity-gpu-locks}"
SEED_LOCK="${SEED_LOCK:-${ROOT}/.offnet42-capacity-recovery.lock}"
CAPACITY_LIMIT="${OFFNET42_CAPACITY_LIMIT:-20000}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

if ! mkdir "${SEED_LOCK}" 2>/dev/null; then
  log "another OFF-Net seed-42 recovery watcher owns ${SEED_LOCK}; exiting"
  exit 0
fi
trap 'rmdir "${SEED_LOCK}" 2>/dev/null || true' EXIT

OUTPUT_DIR="${ROOT}/offnet_seed42"
CHECKPOINT="${OUTPUT_DIR}/best_model.pth"
mkdir -p "${OUTPUT_DIR}" "${GPU_LOCK_ROOT}"
if [[ -s "${OUTPUT_DIR}/result.json" ]]; then
  log "offnet_seed42 already completed; exiting"
  exit 0
fi
if [[ ! -s "${CHECKPOINT}" ]]; then
  log "missing checkpoint ${CHECKPOINT}; cannot resume"
  exit 1
fi

seed_running() {
  pgrep -af '[t]rain_matched_orfd_baseline.py.*--baseline offnet.*--seed 42( |$)' >/dev/null
}

gpu_memory() {
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | tr -d '[:space:]' | head -n 1
}

claim_gpu() {
  local gpu memory
  for gpu in ${GPU_LIST}; do
    if ! mkdir "${GPU_LOCK_ROOT}/gpu${gpu}" 2>/dev/null; then
      continue
    fi
    memory="$(gpu_memory "${gpu}" || true)"
    if [[ "${memory:-999999}" =~ ^[0-9]+$ ]] && (( memory < CAPACITY_LIMIT )); then
      printf '%s\n' "${gpu}"
      return 0
    fi
    rmdir "${GPU_LOCK_ROOT}/gpu${gpu}" 2>/dev/null || true
  done
  return 1
}

release_gpu() {
  rmdir "${GPU_LOCK_ROOT}/gpu$1" 2>/dev/null || true
}

while [[ ! -s "${OUTPUT_DIR}/result.json" ]]; do
  if seed_running; then
    log "offnet_seed42 is already running; waiting for result"
    sleep 60
    continue
  fi

  gpu="$(claim_gpu || true)"
  if [[ -z "${gpu}" ]]; then
    log "no GPU below ${CAPACITY_LIMIT} MiB used; waiting"
    sleep 60
    continue
  fi
  if [[ -s "${OUTPUT_DIR}/result.json" ]] || seed_running; then
    release_gpu "${gpu}"
    continue
  fi

  log "resuming OFF-Net seed-42 on GPU${gpu} from ${CHECKPOINT}"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
    python tools/train_matched_orfd_baseline.py --baseline offnet \
      --official-source "${SOURCE}" --data-root "${ORFD_ROOT}" \
      --normal-root "${NORMAL_ROOT}" --output-dir "${OUTPUT_DIR}" \
      --seed 42 --epochs 30 --resume "${CHECKPOINT}" --batch-size 2 \
      --num-workers 8 --gradient-accumulation-steps 4 --height 704 \
      --width 1280 --val-every 1 --device cuda
  status=$?
  set -e
  release_gpu "${gpu}"
  log "OFF-Net seed-42 process exited with status ${status}"
  [[ -s "${OUTPUT_DIR}/result.json" ]] || sleep 30
done

log "offnet_seed42/result.json detected; recovery watcher exiting"
