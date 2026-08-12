#!/usr/bin/env bash
set -euo pipefail

# Deferred OFF-Net continuation used when GPU1 is temporarily reserved for
# the independent SNE-RoadSeg seed queue.  It preserves the formal OFF-Net
# recipe by resuming only from epoch-boundary checkpoints after all SNE seeds
# have produced result.json.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-1}"
LOG="${LOG:-/tmp/litevilnet_orfd_offnet_after_sne.log}"
DISPATCH_LOG="${DISPATCH_LOG:-/tmp/litevilnet_orfd_offnet_after_sne_dispatch.log}"
LOCK_DIR="${LOCK_DIR:-${ROOT}/.offnet-after-sne-watchdog.lock}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  log "another deferred OFF-Net watchdog owns ${LOCK_DIR}; exiting"
  exit 0
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

for seed in 40 41 42; do
  while [[ ! -s "${ROOT}/sne_roadseg_seed${seed}/result.json" ]]; do
    log "waiting for sne_roadseg_seed${seed}/result.json"
    sleep 60
  done
done

log "all SNE seeds completed; resuming OFF-Net seeds on GPU${GPU}"
setsid env GPU="${GPU}" SEEDS='40 41 42' LOG="${DISPATCH_LOG}" \
  bash tools/dispatch_orfd_offnet_resume.sh
log "deferred OFF-Net continuation completed"
