#!/usr/bin/env bash
set -euo pipefail

# Run the independent SNE-RoadSeg seed-41 continuation alongside seed-40 once
# all GPU1 baselines have released the card.  The primary SNE dispatcher still
# owns seeds 40 and 42; this lane claims only seed 41 and exits safely if
# another process has already claimed or completed it.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-1}"
ENV_NAME="${ENV_NAME:-litevilnet_ral}"
ORFD_ROOT="${ORFD_ROOT:-$PWD/runs/revision_1/matched_orfd/local_data/Final_Dataset}"
NORMAL_ROOT="${NORMAL_ROOT:-$PWD/runs/revision_1/matched_orfd/local_exact_normals/sne_roadseg}"
SOURCE="${SOURCE:-$PWD/third_party/matched_baselines/SNE-RoadSeg}"
LOG="${LOG:-/tmp/litevilnet_orfd_sne41_parallel_after_offnet.log}"
GPU_MEMORY_LIMIT="${GPU_MEMORY_LIMIT:-12000}"
OUTPUT_DIR="${ROOT}/sne_roadseg_seed41"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

for dependency in offnet_seed40 offnet_seed41 offnet_seed42 usnet_seed42; do
  while [[ ! -s "${ROOT}/${dependency}/result.json" ]]; do
    log "waiting for ${dependency}/result.json"
    sleep 60
  done
done

if [[ -s "${OUTPUT_DIR}/result.json" ]]; then
  log "sne_roadseg_seed41 already completed; skipping"
  exit 0
fi
if pgrep -af "train_matched_orfd_baseline.py.*--baseline sne_roadseg.*--seed 41" >/dev/null; then
  log "sne_roadseg_seed41 already running; skipping"
  exit 0
fi
if [[ ! -s "${OUTPUT_DIR}/best_model.pth" ]]; then
  log "missing checkpoint: ${OUTPUT_DIR}/best_model.pth"
  exit 1
fi

while :; do
  memory="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | tr -d '[:space:]' | head -n 1 || true)"
  if [[ "${memory:-999999}" =~ ^[0-9]+$ ]] && (( memory < GPU_MEMORY_LIMIT )); then
    log "GPU${GPU} capacity available (${memory} MiB used; limit ${GPU_MEMORY_LIMIT} MiB)"
    break
  fi
  log "GPU${GPU} busy (${memory:-unknown} MiB used); waiting"
  sleep 60
done

if [[ -s "${OUTPUT_DIR}/result.json" ]] || \
   pgrep -af "train_matched_orfd_baseline.py.*--baseline sne_roadseg.*--seed 41" >/dev/null; then
  log "sne_roadseg_seed41 was claimed while waiting; skipping"
  exit 0
fi

log "resuming SNE-RoadSeg seed-41 on GPU${GPU} from ${OUTPUT_DIR}/best_model.pth"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONDONTWRITEBYTECODE=1 \
  conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
  python tools/train_matched_orfd_baseline.py --baseline sne_roadseg \
    --official-source "${SOURCE}" --data-root "${ORFD_ROOT}" \
    --normal-root "${NORMAL_ROOT}" --output-dir "${OUTPUT_DIR}" \
    --seed 41 --epochs 30 --resume "${OUTPUT_DIR}/best_model.pth" \
    --batch-size 2 --num-workers 8 --height 704 --width 1280 \
    --val-every 1 --amp --device cuda
log "SNE-RoadSeg seed-41 process exited"
