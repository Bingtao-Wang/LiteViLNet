#!/usr/bin/env bash
set -euo pipefail

# Resume the interrupted OFF-Net ORFD seeds from epoch-boundary checkpoints.
# The original protocol is intentionally repeated verbatim: 30 epochs,
# 704x1280 input, batch 2, accumulation 4, and FP32 (no AMP).

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-1}"
ENV_NAME="${ENV_NAME:-litevilnet_roadformer_ral}"
ORFD_ROOT="${ORFD_ROOT:-$PWD/runs/revision_1/matched_orfd/local_data/Final_Dataset}"
NORMAL_ROOT="${NORMAL_ROOT:-$PWD/runs/revision_1/matched_orfd/local_exact_normals/offnet}"
SOURCE="${SOURCE:-$PWD/third_party/matched_baselines/OFF-Net}"
LOG="${LOG:-/tmp/litevilnet_orfd_offnet_resume.log}"
SEEDS="${SEEDS:-40 41 42}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }
wait_capacity() {
  local memory
  while :; do
    memory="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
      | tr -d '[:space:]' | head -n 1 || true)"
    if [[ "${memory:-999999}" =~ ^[0-9]+$ ]] && (( memory < 30000 )); then
      log "GPU${GPU} capacity available (${memory} MiB used; limit 30000 MiB)"
      return 0
    fi
    log "GPU${GPU} busy (${memory:-unknown} MiB used); waiting"
    sleep 60
  done
}

for seed in ${SEEDS}; do
  output_dir="${ROOT}/offnet_seed${seed}"
  if [[ -s "${output_dir}/result.json" ]]; then
    log "offnet_seed${seed} already completed; skipping"
    continue
  fi
  if [[ ! -s "${output_dir}/best_model.pth" ]]; then
    log "missing checkpoint: ${output_dir}/best_model.pth"
    exit 1
  fi
  wait_capacity
  if [[ -s "${output_dir}/result.json" ]]; then
    log "offnet_seed${seed} completed while waiting; skipping"
    continue
  fi
  log "resuming OFF-Net seed-${seed} on GPU${GPU} from ${output_dir}/best_model.pth"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
    python tools/train_matched_orfd_baseline.py --baseline offnet \
      --official-source "${SOURCE}" --data-root "${ORFD_ROOT}" \
      --normal-root "${NORMAL_ROOT}" --output-dir "${output_dir}" \
      --seed "${seed}" --epochs 30 --resume "${output_dir}/best_model.pth" \
      --batch-size 2 --num-workers 8 --gradient-accumulation-steps 4 \
      --height 704 --width 1280 --val-every 1 --device cuda
  log "OFF-Net seed-${seed} process exited"
done
log "all resumable OFF-Net seeds completed"
