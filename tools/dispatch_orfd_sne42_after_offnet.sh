#!/usr/bin/env bash
set -euo pipefail

# Deferred SNE-RoadSeg seed-42 continuation.  Seed 42 has no resumable
# checkpoint, so it is intentionally started from scratch only after the
# higher-priority OFF-Net GPU1 lane has finished.  The training recipe stays
# fixed; this helper changes admission order only.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-1}"
ENV_NAME="${ENV_NAME:-litevilnet_ral}"
ORFD_ROOT="${ORFD_ROOT:-$PWD/runs/revision_1/matched_orfd/local_data/Final_Dataset}"
NORMAL_ROOT="${NORMAL_ROOT:-$PWD/runs/revision_1/matched_orfd/local_exact_normals/sne_roadseg}"
SOURCE="${SOURCE:-$PWD/third_party/matched_baselines/SNE-RoadSeg}"
LOG="${LOG:-/tmp/litevilnet_orfd_sne42_after_offnet.log}"
CAPACITY_LIMIT="${SNE42_CAPACITY_LIMIT:-30000}"
LOCK_DIR="${LOCK_DIR:-${ROOT}/.sne42-after-offnet.lock}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  log "another SNE seed-42 deferred watcher owns ${LOCK_DIR}; exiting"
  exit 0
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

wait_result() {
  local name="$1"
  while [[ ! -s "${ROOT}/${name}/result.json" ]]; do
    log "waiting for ${name}/result.json"
    sleep 60
  done
}

wait_capacity() {
  local memory
  while :; do
    memory="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
      | tr -d '[:space:]' | head -n 1 || true)"
    if [[ "${memory:-999999}" =~ ^[0-9]+$ ]] && (( memory < CAPACITY_LIMIT )); then
      log "GPU${GPU} capacity available (${memory} MiB used; limit ${CAPACITY_LIMIT} MiB)"
      return 0
    fi
    log "GPU${GPU} busy (${memory:-unknown} MiB used); waiting"
    sleep 60
  done
}

if [[ -s "${ROOT}/sne_roadseg_seed42/result.json" ]]; then
  log "sne_roadseg_seed42 already completed; skipping"
  exit 0
fi

# OFF-Net is the priority lane for the current rebuttal snapshot.  Waiting on
# both resumed seeds prevents this fresh high-memory SNE job from competing
# with either OFF-Net continuation.
wait_result offnet_seed41
wait_result offnet_seed42

if pgrep -af "[t]rain_matched_orfd_baseline.py.*--baseline sne_roadseg.*--seed 42( |$)" >/dev/null; then
  log "sne_roadseg_seed42 already running elsewhere; exiting without duplicate"
  exit 0
fi
wait_capacity

output_dir="${ROOT}/sne_roadseg_seed42"
mkdir -p "${output_dir}"
if [[ -s "${output_dir}/result.json" ]]; then
  log "sne_roadseg_seed42 completed while waiting; skipping"
  exit 0
fi
log "launching SNE-RoadSeg seed-42 from scratch on GPU${GPU}"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONDONTWRITEBYTECODE=1 \
  conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
  python tools/train_matched_orfd_baseline.py --baseline sne_roadseg \
    --official-source "${SOURCE}" --data-root "${ORFD_ROOT}" \
    --normal-root "${NORMAL_ROOT}" --output-dir "${output_dir}" \
    --seed 42 --epochs 30 --batch-size 2 --num-workers 8 \
    --height 704 --width 1280 --val-every 1 --amp --device cuda
log "SNE-RoadSeg seed-42 deferred run exited"
