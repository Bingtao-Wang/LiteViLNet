#!/usr/bin/env bash
set -euo pipefail

# Independent follow-up dispatcher for the matched ORFD queue.  It is safe to
# run while the first-wave jobs are active: each follow-up has a marker and is
# launched only after its predecessor has produced result.json.  The memory
# guard leaves the two external Drive-OccWorld jobs untouched.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
LOG="${LOG:-/tmp/litevilnet_orfd_followup_dispatcher.log}"
LOCK_ROOT="${LOCK_ROOT:-${ROOT}/.dispatch-locks}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
MAIN_ENV="${MAIN_ENV:-litevilnet_ral}"
OPENMMLAB_ENV="${OPENMMLAB_ENV:-litevilnet_roadformer_ral}"
ORFD_ROOT="${ORFD_ROOT:-$PWD/runs/revision_1/matched_orfd/local_data/Final_Dataset}"
SNE_NORMAL_ROOT="${SNE_NORMAL_ROOT:-$PWD/runs/revision_1/matched_orfd/local_exact_normals/sne_roadseg}"
OFFNET_NORMAL_ROOT="${OFFNET_NORMAL_ROOT:-$PWD/runs/revision_1/matched_orfd/local_exact_normals/offnet}"
ROADFORMER_ROOT="${ROADFORMER_ROOT:-$PWD/runs/revision_1/matched_orfd/roadformer_orfd_exact}"
USNET_SOURCE="${USNET_SOURCE:-$PWD/third_party/matched_baselines/USNet}"
SNE_SOURCE="${SNE_SOURCE:-$PWD/third_party/matched_baselines/SNE-RoadSeg}"
OFFNET_SOURCE="${OFFNET_SOURCE:-$PWD/third_party/matched_baselines/OFF-Net}"
ROADFORMER_SOURCE="${ROADFORMER_SOURCE:-$PWD/third_party/matched_baselines/Road-Former}"

mkdir -p "${ROOT}" "${LOCK_ROOT}"
exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }
done_file() { [[ -s "${ROOT}/$1/result.json" ]]; }
lock_file() { [[ -e "${LOCK_ROOT}/$1" ]]; }
claim() {
  local name="$1"
  ( set -o noclobber; : >"${LOCK_ROOT}/${name}" ) 2>/dev/null
}
gpu_mem() {
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | tr -d '[:space:]' | head -n 1
}
wait_gpu() {
  local gpu="$1" limit="${2:-30000}" m
  while :; do
    m="$(gpu_mem "${gpu}" || true)"
    if [[ "${m:-999999}" =~ ^[0-9]+$ ]] && (( m < limit )); then
      log "GPU${gpu} capacity available (${m} MiB used; limit ${limit} MiB)"
      return 0
    fi
    log "GPU${gpu} busy (${m:-unknown} MiB used); waiting"
    sleep 60
  done
}
wait_result() {
  local d="$1"
  while ! done_file "${d}"; do sleep 60; done
  log "completed ${d}"
}
run_offnet42() {
  local d=offnet_seed42
  done_file "$d" && return
  claim "$d" || return
  wait_result offnet_seed40
  wait_gpu "$GPU0"
  log "launching OFF-Net seed-42 on GPU${GPU0}"
  CUDA_VISIBLE_DEVICES="${GPU0}" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "${OPENMMLAB_ENV}" env PYTHONPATH=. \
    python tools/train_matched_orfd_baseline.py --baseline offnet \
      --official-source "${OFFNET_SOURCE}" --data-root "${ORFD_ROOT}" \
      --normal-root "${OFFNET_NORMAL_ROOT}" --output-dir "${ROOT}/${d}" \
      --seed 42 --epochs 30 --batch-size 2 --num-workers 8 \
      --gradient-accumulation-steps 4 --height 704 --width 1280 --val-every 1 \
      --device cuda >>/tmp/litevilnet_orfd_offnet42.log 2>&1
  log "OFF-Net seed-42 process exited"
}
run_sne() {
  local seed="$1" d="sne_roadseg_seed${1}"
  done_file "$d" && return
  claim "$d" || return
  wait_result "sne_roadseg_seed$((seed-1))"
  wait_gpu "$GPU1"
  log "launching SNE-RoadSeg seed-${seed} on GPU${GPU1}"
  CUDA_VISIBLE_DEVICES="${GPU1}" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "${MAIN_ENV}" env PYTHONPATH=. \
    python tools/train_matched_orfd_baseline.py --baseline sne_roadseg \
      --official-source "${SNE_SOURCE}" --data-root "${ORFD_ROOT}" \
      --normal-root "${SNE_NORMAL_ROOT}" --output-dir "${ROOT}/${d}" \
      --seed "${seed}" --epochs 30 --batch-size 2 --num-workers 8 \
      --height 704 --width 1280 --val-every 1 --amp --device cuda \
      >>"/tmp/litevilnet_orfd_sne${seed}.log" 2>&1
  log "SNE-RoadSeg seed-${seed} process exited"
}
run_roadformer() {
  local seed="$1" d="roadformer_seed${1}"
  done_file "$d" && return
  claim "$d" || return
  if (( seed == 40 )); then
    wait_result usnet_seed42
    wait_result offnet_seed42
  else
    wait_result "roadformer_seed$((seed-1))"
  fi
  wait_gpu "$GPU0"
  log "launching RoadFormer seed-${seed} on GPU${GPU0}"
  CUDA_VISIBLE_DEVICES="${GPU0}" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "${OPENMMLAB_ENV}" env PYTHONPATH=. \
    python tools/train_matched_orfd_roadformer.py --official-source "${ROADFORMER_SOURCE}" \
      --data-root "${ROADFORMER_ROOT}" --output-dir "${ROOT}/${d}" \
      --seed "${seed}" --epochs 50 --batch-size 4 --num-workers 8 \
      --height 704 --width 1280 --val-every 1 \
      >>"/tmp/litevilnet_orfd_roadformer${seed}.log" 2>&1
  log "RoadFormer seed-${seed} process exited"
}

log 'follow-up dispatcher started; first-wave jobs are not modified'
run_offnet42 &
run_sne 41 &
run_sne 42 &
run_roadformer 40 &
run_roadformer 41 &
run_roadformer 42 &
wait
log 'follow-up dispatcher completed'
