#!/usr/bin/env bash
set -euo pipefail

# Queue the missing SNE-RoadSeg seed-42 run after the checkpoint-based
# seed-40/41 continuation completes.  The training recipe is identical to
# the formal ORFD protocol; this helper only supplies the dependency and
# GPU-capacity guard.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-1}"
ENV_NAME="${ENV_NAME:-litevilnet_ral}"
ORFD_ROOT="${ORFD_ROOT:-$PWD/runs/revision_1/matched_orfd/local_data/Final_Dataset}"
NORMAL_ROOT="${NORMAL_ROOT:-$PWD/runs/revision_1/matched_orfd/local_exact_normals/sne_roadseg}"
SOURCE="${SOURCE:-$PWD/third_party/matched_baselines/SNE-RoadSeg}"
LOG="${LOG:-/tmp/litevilnet_orfd_sne42_after_resume.log}"
GPU_MEMORY_LIMIT="${GPU_MEMORY_LIMIT:-12000}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

while [[ ! -s "${ROOT}/sne_roadseg_seed41/result.json" ]]; do
  log "waiting for sne_roadseg_seed41/result.json"
  sleep 60
done

if [[ -s "${ROOT}/sne_roadseg_seed42/result.json" ]]; then
  log "sne_roadseg_seed42 already completed; skipping"
  exit 0
fi
if pgrep -af "train_matched_orfd_baseline.py.*--baseline sne_roadseg.*--seed 42" >/dev/null; then
  log "sne_roadseg seed-42 is already running; stopping continuation"
  exit 0
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

output_dir="${ROOT}/sne_roadseg_seed42"
mkdir -p "${output_dir}"
log "launching SNE-RoadSeg seed-42 from scratch on GPU${GPU}"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONDONTWRITEBYTECODE=1 \
  conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
  python tools/train_matched_orfd_baseline.py --baseline sne_roadseg \
    --official-source "${SOURCE}" --data-root "${ORFD_ROOT}" \
    --normal-root "${NORMAL_ROOT}" --output-dir "${output_dir}" \
    --seed 42 --epochs 30 --batch-size 2 --num-workers 8 \
    --height 704 --width 1280 --val-every 1 --amp --device cuda
log "SNE-RoadSeg seed-42 process exited"
