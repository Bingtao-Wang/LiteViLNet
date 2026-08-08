#!/usr/bin/env bash
set -euo pipefail

# SNE-RoadSeg is intentionally deferred while both GPU slots are used by
# OFF-Net and RoadFormer.  Each saved best_model.pth is an epoch-boundary
# checkpoint; this continuation restores all three seeds with exactly the
# original ORFD protocol once RoadFormer has released GPU0.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-0}"
ENV_NAME="${ENV_NAME:-litevilnet_ral}"
ORFD_ROOT="${ORFD_ROOT:-$PWD/runs/revision_1/matched_orfd/local_data/Final_Dataset}"
NORMAL_ROOT="${NORMAL_ROOT:-$PWD/runs/revision_1/matched_orfd/local_exact_normals/sne_roadseg}"
SOURCE="${SOURCE:-$PWD/third_party/matched_baselines/SNE-RoadSeg}"
LOG="${LOG:-/tmp/litevilnet_orfd_sne40_deferred.log}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

while [[ ! -s "${ROOT}/roadformer_seed42/result.json" ]]; do
  log "waiting for RoadFormer seed-42 before resuming deferred SNE seeds"
  sleep 60
done

for seed in 40 41 42; do
  output_dir="${ROOT}/sne_roadseg_seed${seed}"
  if [[ -s "${output_dir}/result.json" ]]; then
    log "sne_roadseg_seed${seed} already completed; skipping"
    continue
  fi
  if [[ ! -s "${output_dir}/best_model.pth" ]]; then
    log "missing deferred checkpoint: ${output_dir}/best_model.pth"
    exit 1
  fi
  if pgrep -af "train_matched_orfd_baseline.py.*--baseline sne_roadseg.*--seed ${seed}" >/dev/null; then
    log "sne_roadseg seed-${seed} is already running; stopping continuation"
    exit 0
  fi
  log "resuming SNE-RoadSeg seed-${seed} on GPU${GPU} from its saved checkpoint"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
    python tools/train_matched_orfd_baseline.py --baseline sne_roadseg \
      --official-source "${SOURCE}" --data-root "${ORFD_ROOT}" \
      --normal-root "${NORMAL_ROOT}" --output-dir "${output_dir}" \
      --seed "${seed}" --epochs 30 --resume "${output_dir}/best_model.pth" \
      --batch-size 2 --num-workers 8 --height 704 --width 1280 \
      --val-every 1 --amp --device cuda \
      >>"/tmp/litevilnet_orfd_sne${seed}_deferred_train.log" 2>&1
  log "deferred SNE-RoadSeg seed-${seed} process exited"
done
log "all deferred SNE-RoadSeg seeds completed"
