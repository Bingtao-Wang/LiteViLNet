#!/usr/bin/env bash
set -euo pipefail

# Wait for the complete formal ORFD seed matrix, then run the strict
# provenance/protocol checker and write the canonical multi-method summary.
# This monitor never starts or changes a training job.

ROOT="${ROOT:-runs/revision_1/matched_orfd/formal}"
ENV_NAME="${ENV_NAME:-litevilnet_ral}"
LOG="${LOG:-/tmp/litevilnet_orfd_final_summary.log}"
OUTPUT_JSON="${OUTPUT_JSON:-docs/ral/orfd_matched_baselines/results/summary.json}"
OUTPUT_CSV="${OUTPUT_CSV:-docs/ral/orfd_matched_baselines/results/summary.csv}"
SEED_OUTPUT_DIR="${SEED_OUTPUT_DIR:-docs/ral/orfd_matched_baselines/results/seeds}"

exec >>"${LOG}" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

for method in usnet sne_roadseg offnet roadformer; do
  for seed in 40 41 42; do
    while [[ ! -s "${ROOT}/${method}_seed${seed}/result.json" ]]; do
      log "waiting for ${method}_seed${seed}/result.json"
      sleep 60
    done
  done
done

log "all formal ORFD results are present; running strict summarizer"
conda run --no-capture-output -n "${ENV_NAME}" python \
  tools/summarize_matched_orfd_baselines.py \
  --input-root "${ROOT}" --expected-seeds 40,41,42 \
  --output-json "${OUTPUT_JSON}" --output-csv "${OUTPUT_CSV}" \
  --seed-output-dir "${SEED_OUTPUT_DIR}"
log "canonical ORFD summary completed"
