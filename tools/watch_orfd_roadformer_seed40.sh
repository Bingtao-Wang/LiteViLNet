#!/usr/bin/env bash
set -euo pipefail

# Resilient continuation for the long RoadFormer seed-40 lane.  It is a
# watchdog only: while the official process is alive it does nothing; if the
# process exits before result.json is written, it resumes the latest
# epoch-boundary checkpoint with the unchanged formal recipe.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-0}"
ENV_NAME="${ENV_NAME:-litevilnet_roadformer_ral}"
ROADFORMER_ROOT="${ROADFORMER_ROOT:-$PWD/runs/revision_1/matched_orfd/roadformer_orfd_exact}"
ROADFORMER_SOURCE="${ROADFORMER_SOURCE:-$PWD/third_party/matched_baselines/Road-Former}"
OUTPUT_DIR="${ROOT}/roadformer_seed40"
LOG="${LOG:-/tmp/litevilnet_orfd_roadformer40_watchdog.log}"
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:32}"
NUM_WORKERS="${NUM_WORKERS:-2}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

while [[ ! -s "${OUTPUT_DIR}/result.json" ]]; do
  if pgrep -af "[t]rain_matched_orfd_roadformer.py.*roadformer_seed40" >/dev/null; then
    log "RoadFormer seed-40 is running; watchdog sleeping"
    sleep 60
    continue
  fi

  checkpoint="${OUTPUT_DIR}/best_model.pth"
  if [[ -s "${checkpoint}" ]]; then
    log "RoadFormer seed-40 exited without result; resuming ${checkpoint}"
    if ! CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}" PYTHONDONTWRITEBYTECODE=1 \
      conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
      python tools/train_matched_orfd_roadformer.py \
        --official-source "${ROADFORMER_SOURCE}" \
        --data-root "${ROADFORMER_ROOT}" --output-dir "${OUTPUT_DIR}" \
        --seed 40 --epochs 50 --resume "${checkpoint}" \
        --batch-size 4 --num-workers "${NUM_WORKERS}" --height 704 --width 1280 \
        --val-every 1; then
      log "RoadFormer seed-40 resume exited non-zero; retrying from checkpoint after 30 seconds"
      sleep 30
    fi
  else
    log "RoadFormer seed-40 exited before a checkpoint; restarting formal recipe"
    if ! CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}" PYTHONDONTWRITEBYTECODE=1 \
      conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
      python tools/train_matched_orfd_roadformer.py \
        --official-source "${ROADFORMER_SOURCE}" \
        --data-root "${ROADFORMER_ROOT}" --output-dir "${OUTPUT_DIR}" \
        --seed 40 --epochs 50 --batch-size 4 --num-workers "${NUM_WORKERS}" \
        --height 704 --width 1280 --val-every 1; then
      log "RoadFormer seed-40 restart exited non-zero; retrying after 30 seconds"
      sleep 30
    fi
  fi
done

log "RoadFormer seed-40 result.json detected; watchdog exiting"
