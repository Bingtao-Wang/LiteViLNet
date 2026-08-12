#!/usr/bin/env bash
set -euo pipefail

# Recovery-only dispatcher for SNE-RoadSeg seed 40.  The previous worker
# reached an epoch-boundary checkpoint and exited before writing result.json.
# This helper changes admission/retry scheduling only; the formal recipe is
# repeated verbatim (30 epochs, batch 2, AMP, 704x1280).

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
# Keep GPU0 reserved for the high-memory RoadFormer lane.  Operators can
# still override GPU_LIST explicitly when scheduling on a different host.
GPU_LIST="${GPU_LIST:-1}"
ENV_NAME="${ENV_NAME:-litevilnet_ral}"
ORFD_ROOT="${ORFD_ROOT:-$PWD/runs/revision_1/matched_orfd/local_data/Final_Dataset}"
NORMAL_ROOT="${NORMAL_ROOT:-$PWD/runs/revision_1/matched_orfd/local_exact_normals/sne_roadseg}"
SOURCE="${SOURCE:-$PWD/third_party/matched_baselines/SNE-RoadSeg}"
LOG="${LOG:-/tmp/litevilnet_orfd_sne40_capacity_recovery.log}"
GPU_LOCK_ROOT="${GPU_LOCK_ROOT:-${ROOT}/.sne-capacity-gpu-locks}"
SEED_LOCK="${SEED_LOCK:-${ROOT}/.sne40-capacity-recovery.lock}"
# Scheduling guard: permit admission alongside one OFF-Net worker, but not
# when the card is already carrying two heavy workers.
CAPACITY_LIMIT="${SNE_CAPACITY_LIMIT:-20000}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

if ! mkdir "${SEED_LOCK}" 2>/dev/null; then
  log "another SNE seed-40 recovery watcher owns ${SEED_LOCK}; exiting"
  exit 0
fi
trap 'rmdir "${SEED_LOCK}" 2>/dev/null || true' EXIT

OUTPUT_DIR="${ROOT}/sne_roadseg_seed40"
CHECKPOINT="${OUTPUT_DIR}/best_model.pth"
mkdir -p "${OUTPUT_DIR}" "${GPU_LOCK_ROOT}"
if [[ -s "${OUTPUT_DIR}/result.json" ]]; then
  log "sne_roadseg_seed40 already completed; exiting"
  exit 0
fi
if [[ ! -s "${CHECKPOINT}" ]]; then
  log "missing checkpoint ${CHECKPOINT}; cannot resume"
  exit 1
fi

seed_running() {
  pgrep -af '[t]rain_matched_orfd_baseline.py.*--baseline sne_roadseg.*--seed 40( |$)' >/dev/null
}

gpu_memory() {
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | tr -d '[:space:]' | head -n 1
}

claim_gpu() {
  local gpu memory
  for gpu in ${GPU_LIST}; do
    # Share the lock convention with the multi-seed SNE queue so two SNE
    # workers cannot admit themselves to the same card.
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
    log "sne_roadseg_seed40 is already running; waiting for result"
    sleep 60
    continue
  fi

  gpu="$(claim_gpu || true)"
  if [[ -z "${gpu}" ]]; then
    log "no SNE GPU slot below ${CAPACITY_LIMIT} MiB; waiting"
    sleep 60
    continue
  fi
  if [[ -s "${OUTPUT_DIR}/result.json" ]] || seed_running; then
    release_gpu "${gpu}"
    continue
  fi

  log "resuming SNE-RoadSeg seed-40 on GPU${gpu} from ${CHECKPOINT}"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
    python tools/train_matched_orfd_baseline.py --baseline sne_roadseg \
      --official-source "${SOURCE}" --data-root "${ORFD_ROOT}" \
      --normal-root "${NORMAL_ROOT}" --output-dir "${OUTPUT_DIR}" \
      --seed 40 --epochs 30 --resume "${CHECKPOINT}" --batch-size 2 \
      --num-workers 8 --height 704 --width 1280 --val-every 1 --amp --device cuda
  status=$?
  set -e
  release_gpu "${gpu}"
  log "SNE-RoadSeg seed-40 process exited with status ${status}"
  [[ -s "${OUTPUT_DIR}/result.json" ]] || sleep 30
done

log "sne_roadseg_seed40/result.json detected; recovery watcher exiting"
