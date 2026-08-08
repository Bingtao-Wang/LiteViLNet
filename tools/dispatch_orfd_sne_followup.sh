#!/usr/bin/env bash
set -euo pipefail

# Capacity-aware SNE-RoadSeg continuation.  SNE seed-40 is already running on
# GPU0.  Seeds 41 and 42 are independent and use the unchanged official
# training recipe; placing them sequentially on GPU1 overlaps them with the
# two smaller OFF-Net jobs without launching duplicate processes.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-1}"
ENV_NAME="${ENV_NAME:-litevilnet_ral}"
ORFD_ROOT="${ORFD_ROOT:-$PWD/runs/revision_1/matched_orfd/local_data/Final_Dataset}"
NORMAL_ROOT="${NORMAL_ROOT:-$PWD/runs/revision_1/matched_orfd/local_exact_normals/sne_roadseg}"
SOURCE="${SOURCE:-$PWD/third_party/matched_baselines/SNE-RoadSeg}"
LOG="${LOG:-/tmp/litevilnet_orfd_sne_dispatcher.log}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }
gpu_mem() {
  nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits \
    | tr -d '[:space:]' | head -n 1
}
wait_capacity() {
  local limit="${1:-32000}" memory
  while :; do
    memory="$(gpu_mem || true)"
    if [[ "${memory:-999999}" =~ ^[0-9]+$ ]] && (( memory < limit )); then
      log "GPU${GPU} capacity available (${memory} MiB used; limit ${limit} MiB)"
      return
    fi
    log "GPU${GPU} busy (${memory:-unknown} MiB used); waiting"
    sleep 60
  done
}

for seed in 41 42; do
  output_dir="${ROOT}/sne_roadseg_seed${seed}"
  if [[ -s "${output_dir}/result.json" ]]; then
    log "skipping completed sne_roadseg_seed${seed}"
    continue
  fi
  if pgrep -af "train_matched_orfd_baseline.py.*--baseline sne_roadseg.*--seed ${seed}" >/dev/null; then
    log "sne_roadseg_seed${seed} is already running; stopping continuation"
    exit 0
  fi
  wait_capacity 32000
  log "launching SNE-RoadSeg seed-${seed} on GPU${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
    python tools/train_matched_orfd_baseline.py --baseline sne_roadseg \
      --official-source "${SOURCE}" --data-root "${ORFD_ROOT}" \
      --normal-root "${NORMAL_ROOT}" --output-dir "${output_dir}" \
      --seed "${seed}" --epochs 30 --batch-size 2 --num-workers 8 \
      --height 704 --width 1280 --val-every 1 --amp --device cuda \
      >>"/tmp/litevilnet_orfd_sne${seed}_gpu1.log" 2>&1
  log "SNE-RoadSeg seed-${seed} process exited"
done
log "SNE-RoadSeg continuation completed"
