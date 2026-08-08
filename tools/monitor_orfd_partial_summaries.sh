#!/usr/bin/env bash
set -euo pipefail

# Produce independently auditable per-method summaries as soon as all three
# seeds of a method finish.  The final all-method monitor remains responsible
# for summary.json/summary.csv; these files are deliberately separate.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
LOG="${LOG:-/tmp/litevilnet_orfd_partial_summary.log}"
OUT="${OUT:-docs/ral/orfd_matched_baselines/results}"
ENV_NAME="${ENV_NAME:-litevilnet_ral}"
mkdir -p "${OUT}"
exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

summarize_method() {
  local method="$1"
  local json_out="${OUT}/summary_${method}.json"
  local csv_out="${OUT}/summary_${method}.csv"
  local seed_out="${OUT}/seeds_${method}"
  while :; do
    local missing=0
    for seed in 40 41 42; do
      [[ -s "${ROOT}/${method}_seed${seed}/result.json" ]] || missing=$((missing + 1))
    done
    if (( missing == 0 )); then
      log "all ${method} seeds present; running strict summarizer"
      conda run --no-capture-output -n "${ENV_NAME}" python \
        tools/summarize_matched_orfd_baselines.py --input-root "${ROOT}" \
        --expected-seeds 40,41,42 --methods "${method}" \
        --output-json "${json_out}" --output-csv "${csv_out}" \
        --seed-output-dir "${seed_out}"
      log "${method} strict summary completed"
      return
    fi
    log "waiting for ${method}: ${missing} result files missing"
    sleep 60
  done
}

# Keep a per-method audit available as soon as each independent three-seed
# queue finishes.  The all-method monitor still owns the canonical summary;
# these files are method-scoped snapshots for reproducibility and rebuttal
# bookkeeping.
for method in usnet sne_roadseg offnet roadformer; do
  summarize_method "${method}" &
done
wait
log 'partial method summaries completed'
