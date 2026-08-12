#!/usr/bin/env bash
set -euo pipefail

# Capacity-aware continuation for the interrupted USNet seed-42 run.  The
# checkpoint was written at an epoch boundary; resuming it preserves the
# original 30-epoch protocol and does not change any training hyperparameter.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
# USNet's AMP batch-2 run is the small (~4 GiB) job; keep it on GPU1 so that
# the high-resolution RoadFormer run can use GPU0 without contention.
GPU="${GPU:-1}"
ENV_NAME="${ENV_NAME:-litevilnet_ral}"
ORFD_ROOT="${ORFD_ROOT:-$PWD/runs/revision_1/matched_orfd/local_data/Final_Dataset}"
NORMAL_ROOT="${NORMAL_ROOT:-$PWD/runs/revision_1/matched_orfd/local_exact_normals/sne_roadseg}"
SOURCE="${SOURCE:-$PWD/third_party/matched_baselines/USNet}"
OUTPUT_DIR="${ROOT}/usnet_seed42"
LOG="${LOG:-/tmp/litevilnet_orfd_usnet42_resume.log}"
GPU_MEMORY_LIMIT="${GPU_MEMORY_LIMIT:-45000}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

if [[ -s "${OUTPUT_DIR}/result.json" ]]; then
  log "usnet_seed42 already completed; skipping"
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

log "resuming USNet seed-42 on GPU${GPU} from ${OUTPUT_DIR}/best_model.pth"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONDONTWRITEBYTECODE=1 \
  conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
  python tools/train_matched_orfd_baseline.py --baseline usnet \
    --official-source "${SOURCE}" --data-root "${ORFD_ROOT}" \
    --normal-root "${NORMAL_ROOT}" --output-dir "${OUTPUT_DIR}" \
    --seed 42 --epochs 30 --resume "${OUTPUT_DIR}/best_model.pth" \
    --batch-size 2 --num-workers 8 --height 704 --width 1280 \
    --val-every 1 --amp --device cuda
log "USNet seed-42 process exited"
