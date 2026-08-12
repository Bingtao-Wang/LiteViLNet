#!/usr/bin/env bash
set -euo pipefail

# Resilient watchdog for the sequential OFF-Net ORFD lane.  It never changes
# the formal recipe; it only re-enters the existing checkpoint-based
# dispatcher if an external interruption leaves it without a live process.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
GPU="${GPU:-1}"
LOG="${LOG:-/tmp/litevilnet_orfd_offnet_sequential_watchdog.log}"
DISPATCH_LOG="${DISPATCH_LOG:-/tmp/litevilnet_orfd_offnet_sequential_resume.log}"
LOCK_DIR="${LOCK_DIR:-${ROOT}/.offnet-sequential-watchdog.lock}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  log "another sequential OFF-Net watchdog owns ${LOCK_DIR}; exiting"
  exit 0
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

has_all_results() {
  for seed in 40 41 42; do
    [[ -s "${ROOT}/offnet_seed${seed}/result.json" ]] || return 1
  done
}

while ! has_all_results; do
  if pgrep -af '[d]ispatch_orfd_offnet_resume\.sh|[t]rain_matched_orfd_baseline\.py.*--baseline offnet' >/dev/null; then
    sleep 60
    continue
  fi

  log "sequential OFF-Net lane is absent; restarting checkpoint dispatcher"
  if ! setsid env GPU="${GPU}" SEEDS='40 41 42' LOG="${DISPATCH_LOG}" \
      bash tools/dispatch_orfd_offnet_resume.sh; then
    log "checkpoint dispatcher exited non-zero; retrying after 60 seconds"
    sleep 60
  fi
done

log 'all sequential OFF-Net result files detected; watchdog exiting'
