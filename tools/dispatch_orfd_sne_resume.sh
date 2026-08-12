#!/usr/bin/env bash
set -euo pipefail

# Resume the interrupted SNE-RoadSeg ORFD seeds after the other GPU1 jobs have
# released the card.  The method-specific recipe is unchanged: 30 epochs,
# batch 2, AMP, and 704x1280 input.  SNE-RoadSeg's measured peak allocation is
# about 34.7 GiB on the matched setup, so it must not overlap an OFF-Net
# process merely because the instantaneous free-memory check looks sufficient.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-1}"
ENV_NAME="${ENV_NAME:-litevilnet_ral}"
ORFD_ROOT="${ORFD_ROOT:-$PWD/runs/revision_1/matched_orfd/local_data/Final_Dataset}"
NORMAL_ROOT="${NORMAL_ROOT:-$PWD/runs/revision_1/matched_orfd/local_exact_normals/sne_roadseg}"
SOURCE="${SOURCE:-$PWD/third_party/matched_baselines/SNE-RoadSeg}"
LOG="${LOG:-/tmp/litevilnet_orfd_sne_resume.log}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }
wait_result() {
  local name="$1"
  while [[ ! -s "${ROOT}/${name}/result.json" ]]; do
    log "waiting for ${name}/result.json"
    sleep 60
  done
  log "found completed ${name}"
}
wait_capacity() {
  local memory
  while :; do
    memory="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
      | tr -d '[:space:]' | head -n 1 || true)"
    if [[ "${memory:-999999}" =~ ^[0-9]+$ ]] && (( memory < 12000 )); then
      log "GPU${GPU} capacity available (${memory} MiB used; limit 30000 MiB)"
      return 0
    fi
    log "GPU${GPU} busy (${memory:-unknown} MiB used); waiting"
    sleep 60
  done
}

# Wait for every GPU1 baseline and the small USNet continuation to release
# their allocations.  This is a scheduling guard only; it does not alter the
# fixed training protocol of any method.
for dependency in offnet_seed40 offnet_seed41 offnet_seed42 usnet_seed42; do
  wait_result "${dependency}"
done
for seed in 40 41 42; do
  output_dir="${ROOT}/sne_roadseg_seed${seed}"
  if [[ -s "${output_dir}/result.json" ]]; then
    log "sne_roadseg_seed${seed} already completed; skipping"
    continue
  fi
  if [[ ! -s "${output_dir}/best_model.pth" ]]; then
    # Seed-42 is intentionally owned by dispatch_orfd_sne42_after_resume.sh;
    # leave it for that queue instead of treating its absent checkpoint as a
    # failure of the resumable seed-40/41 lane.
    log "no resumable checkpoint for sne_roadseg_seed${seed}; leaving it to its dedicated queue"
    continue
  fi
  if pgrep -af "train_matched_orfd_baseline.py.*--baseline sne_roadseg.*--seed ${seed}" >/dev/null; then
    log "sne_roadseg_seed${seed} already running; skipping duplicate launch"
    continue
  fi
  wait_capacity
  if [[ -s "${output_dir}/result.json" ]]; then
    log "sne_roadseg_seed${seed} completed while waiting; skipping"
    continue
  fi
  if pgrep -af "train_matched_orfd_baseline.py.*--baseline sne_roadseg.*--seed ${seed}" >/dev/null; then
    log "sne_roadseg_seed${seed} was claimed while waiting; skipping duplicate launch"
    continue
  fi
  log "resuming SNE-RoadSeg seed-${seed} on GPU${GPU} from ${output_dir}/best_model.pth"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "${ENV_NAME}" env PYTHONPATH=. \
    python tools/train_matched_orfd_baseline.py --baseline sne_roadseg \
      --official-source "${SOURCE}" --data-root "${ORFD_ROOT}" \
      --normal-root "${NORMAL_ROOT}" --output-dir "${output_dir}" \
      --seed "${seed}" --epochs 30 --resume "${output_dir}/best_model.pth" \
      --batch-size 2 --num-workers 8 --height 704 --width 1280 \
      --val-every 1 --amp --device cuda
  log "SNE-RoadSeg seed-${seed} process exited"
done
log "all resumable SNE-RoadSeg seeds completed"
