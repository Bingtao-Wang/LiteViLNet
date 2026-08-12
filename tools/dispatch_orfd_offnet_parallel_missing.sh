#!/usr/bin/env bash
set -euo pipefail

# Capacity-controlled continuation for the two missing OFF-Net seeds.  This
# scheduler only changes when jobs are launched; each training invocation
# keeps the formal ORFD recipe unchanged (30 epochs, 704x1280, batch 2,
# accumulation 4, FP32).  It allows at most two OFF-Net processes at once;
# the third seed is claimed only after one of the first two writes its result.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-1}"
ENV_NAME="${ENV_NAME:-litevilnet_roadformer_ral}"
ORFD_ROOT="${ORFD_ROOT:-$PWD/runs/revision_1/matched_orfd/local_data/Final_Dataset}"
NORMAL_ROOT="${NORMAL_ROOT:-$PWD/runs/revision_1/matched_orfd/local_exact_normals/offnet}"
SOURCE="${SOURCE:-$PWD/third_party/matched_baselines/OFF-Net}"
LOG="${LOG:-/tmp/litevilnet_orfd_offnet_parallel_missing.log}"
MEMORY_LIMIT="${MEMORY_LIMIT:-30000}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

gpu_mem() {
  nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | tr -d '[:space:]' | head -n 1
}

wait_capacity() {
  local memory
  while :; do
    memory="$(gpu_mem || true)"
    if [[ "${memory:-999999}" =~ ^[0-9]+$ ]] && (( memory < MEMORY_LIMIT )); then
      log "GPU${GPU} capacity available (${memory} MiB used; limit ${MEMORY_LIMIT} MiB)"
      return 0
    fi
    log "GPU${GPU} busy (${memory:-unknown} MiB used); waiting"
    sleep 60
  done
}

seed_running() {
  local seed="$1"
  pgrep -af "train_matched_orfd_baseline.py.*--baseline offnet.*--seed ${seed}( |$)" >/dev/null
}

launch_seed() {
  local seed="$1"
  local output_dir="${ROOT}/offnet_seed${seed}"
  if [[ -s "${output_dir}/result.json" ]]; then
    log "offnet_seed${seed} already completed; skipping"
    return 0
  fi
  if seed_running "${seed}"; then
    log "offnet_seed${seed} already running; skipping duplicate launch"
    return 0
  fi
  if [[ ! -s "${output_dir}/best_model.pth" ]]; then
    log "missing checkpoint: ${output_dir}/best_model.pth"
    return 1
  fi
  log "resuming OFF-Net seed-${seed} on GPU${GPU} from ${output_dir}/best_model.pth"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
    python tools/train_matched_orfd_baseline.py --baseline offnet \
      --official-source "${SOURCE}" --data-root "${ORFD_ROOT}" \
      --normal-root "${NORMAL_ROOT}" --output-dir "${output_dir}" \
      --seed "${seed}" --epochs 30 --resume "${output_dir}/best_model.pth" \
      --batch-size 2 --num-workers 8 --gradient-accumulation-steps 4 \
      --height 704 --width 1280 --val-every 1 --device cuda \
      >>"/tmp/litevilnet_orfd_offnet_seed${seed}_parallel.log" 2>&1 &
  echo $!
}

# Seed 40 is owned by the existing checkpoint-resume process.  Start seed 41
# as soon as the memory guard permits, then wait for seed 40 or 41 to finish
# before starting seed 42.  Waiting on a result (rather than an instantaneous
# memory sample) avoids a launch race while the second process is allocating.
wait_capacity
launch_seed 41 || true

while :; do
  if [[ -s "${ROOT}/offnet_seed40/result.json" ]] || \
     [[ -s "${ROOT}/offnet_seed41/result.json" ]]; then
    break
  fi
  log "waiting for seed40 or seed41 to finish before launching seed42"
  sleep 60
done
launch_seed 42 || true
log "parallel OFF-Net continuation has claimed all missing seeds"
