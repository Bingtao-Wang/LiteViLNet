#!/usr/bin/env bash
set -euo pipefail

# Deferred SNE-RoadSeg seed-40 continuation. It resumes the saved epoch-4
# checkpoint after the priority OFF-Net seed-40 GPU0 lane completes. Only
# scheduling order changes; the formal 30-epoch AMP recipe is unchanged.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-0}"
ENV_NAME="${ENV_NAME:-litevilnet_ral}"
ORFD_ROOT="${ORFD_ROOT:-$PWD/runs/revision_1/matched_orfd/local_data/Final_Dataset}"
NORMAL_ROOT="${NORMAL_ROOT:-$PWD/runs/revision_1/matched_orfd/local_exact_normals/sne_roadseg}"
SOURCE="${SOURCE:-$PWD/third_party/matched_baselines/SNE-RoadSeg}"
LOG="${LOG:-/tmp/litevilnet_orfd_sne40_after_offnet.log}"
CAPACITY_LIMIT="${SNE40_CAPACITY_LIMIT:-30000}"
LOCK_DIR="${LOCK_DIR:-${ROOT}/.sne40-after-offnet.lock}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  log "another SNE seed-40 deferred watcher owns ${LOCK_DIR}; exiting"
  exit 0
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

while [[ ! -s "${ROOT}/offnet_seed40/result.json" ]]; do
  log "waiting for offnet_seed40/result.json"
  sleep 60
done

if [[ -s "${ROOT}/sne_roadseg_seed40/result.json" ]]; then
  log "sne_roadseg_seed40 already completed; skipping"
  exit 0
fi
if pgrep -af "[t]rain_matched_orfd_baseline.py.*--baseline sne_roadseg.*--seed 40( |$)" >/dev/null; then
  log "sne_roadseg seed-40 is already running; exiting without duplicate"
  exit 0
fi

while :; do
  memory="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | tr -d '[:space:]' | head -n 1 || true)"
  if [[ "${memory:-999999}" =~ ^[0-9]+$ ]] && (( memory < CAPACITY_LIMIT )); then
    log "GPU${GPU} capacity available (${memory} MiB used; limit ${CAPACITY_LIMIT} MiB)"
    break
  fi
  log "GPU${GPU} busy (${memory:-unknown} MiB used); waiting"
  sleep 60
done

output_dir="${ROOT}/sne_roadseg_seed40"
if [[ ! -s "${output_dir}/best_model.pth" ]]; then
  log "missing SNE seed-40 checkpoint: ${output_dir}/best_model.pth"
  exit 1
fi
log "resuming SNE-RoadSeg seed-40 on GPU${GPU} from ${output_dir}/best_model.pth"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONDONTWRITEBYTECODE=1 \
  conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
  python tools/train_matched_orfd_baseline.py --baseline sne_roadseg \
    --official-source "${SOURCE}" --data-root "${ORFD_ROOT}" \
    --normal-root "${NORMAL_ROOT}" --output-dir "${output_dir}" \
    --seed 40 --epochs 30 --resume "${output_dir}/best_model.pth" \
    --batch-size 2 --num-workers 8 --height 704 --width 1280 \
    --val-every 1 --amp --device cuda
log "SNE-RoadSeg seed-40 deferred run exited"
