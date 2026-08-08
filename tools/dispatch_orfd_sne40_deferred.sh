#!/usr/bin/env bash
set -euo pipefail

# SNE-RoadSeg seed-40 is intentionally deferred while the GPU0 slot is used
# by OFF-Net and RoadFormer.  The saved best_model.pth is an epoch-boundary
# checkpoint; this continuation restores it with exactly the original ORFD
# protocol once RoadFormer has released GPU0.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-0}"
ENV_NAME="${ENV_NAME:-litevilnet_ral}"
ORFD_ROOT="${ORFD_ROOT:-$PWD/runs/revision_1/matched_orfd/local_data/Final_Dataset}"
NORMAL_ROOT="${NORMAL_ROOT:-$PWD/runs/revision_1/matched_orfd/local_exact_normals/sne_roadseg}"
SOURCE="${SOURCE:-$PWD/third_party/matched_baselines/SNE-RoadSeg}"
LOG="${LOG:-/tmp/litevilnet_orfd_sne40_deferred.log}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

if [[ -s "${ROOT}/sne_roadseg_seed40/result.json" ]]; then
  log "sne_roadseg_seed40 already completed; nothing to do"
  exit 0
fi

while [[ ! -s "${ROOT}/roadformer_seed42/result.json" ]]; do
  log "waiting for RoadFormer seed-42 before resuming SNE seed-40"
  sleep 60
done

if pgrep -af "train_matched_orfd_baseline.py.*--baseline sne_roadseg.*--seed 40" >/dev/null; then
  log "sne_roadseg seed-40 is already running; nothing to do"
  exit 0
fi

if [[ ! -s "${ROOT}/sne_roadseg_seed40/best_model.pth" ]]; then
  log "missing deferred checkpoint: ${ROOT}/sne_roadseg_seed40/best_model.pth"
  exit 1
fi

log "resuming SNE-RoadSeg seed-40 on GPU${GPU} from the saved checkpoint"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONDONTWRITEBYTECODE=1 \
  conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
  python tools/train_matched_orfd_baseline.py --baseline sne_roadseg \
    --official-source "${SOURCE}" --data-root "${ORFD_ROOT}" \
    --normal-root "${NORMAL_ROOT}" --output-dir "${ROOT}/sne_roadseg_seed40" \
    --seed 40 --epochs 30 --resume "${ROOT}/sne_roadseg_seed40/best_model.pth" \
    --batch-size 2 --num-workers 8 --height 704 --width 1280 \
    --val-every 1 --amp --device cuda \
    >>"/tmp/litevilnet_orfd_sne40_deferred_train.log" 2>&1
log "deferred SNE-RoadSeg seed-40 process exited"
