#!/usr/bin/env bash
set -euo pipefail

# Start the high-memory RoadFormer queue as soon as USNet-42 releases GPU0.
# OFF-Net is independent on GPU1, so waiting for it would leave GPU0 idle.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
LOG="${LOG:-/tmp/litevilnet_orfd_roadformer_after_usnet.log}"
exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

while [[ ! -s "${ROOT}/usnet_seed42/result.json" ]]; do
  log "waiting for usnet_seed42/result.json before starting RoadFormer"
  sleep 60
done
log "USNet-42 completed; starting RoadFormer queue"
WAIT_FOR_OFFNET=0 LOG="${LOG}" bash tools/dispatch_orfd_roadformer_after_sne.sh
log "RoadFormer queue completed"
