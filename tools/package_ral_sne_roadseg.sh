#!/usr/bin/env bash
set -euo pipefail

# Build only the completed anonymous SNE-RoadSeg ORFD supplement. Datasets,
# checkpoints, generated caches, and third-party source trees are excluded.

OUTPUT="${1:-dist/LiteViLNet_RAL_SNE_ORFD_Anonymous_Reproduction.tar.gz}"
RESULT_ROOT="runs/revision_1/matched_orfd/formal"
SUMMARY_ROOT="docs/ral/orfd_matched_baselines/results"

required=(
  "docs/ral/sne_roadseg_orfd/README.md"
  "docs/ral/sne_roadseg_orfd/source_provenance.json"
  "docs/ral/sne_roadseg_orfd/RESULTS.md"
  "${SUMMARY_ROOT}/summary_sne_roadseg.json"
  "${SUMMARY_ROOT}/summary_sne_roadseg.csv"
)
for seed in 40 41 42; do
  required+=("${RESULT_ROOT}/sne_roadseg_seed${seed}/result.json")
done
for path in "${required[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "Missing required SNE evidence: $path" >&2
    exit 1
  fi
done

staging="$(mktemp -d)"
raw="$(mktemp -d)"
trap 'rm -rf "$staging" "$raw"' EXIT

copy_file() {
  local source="$1" destination="$2"
  mkdir -p "$staging/$(dirname "$destination")"
  cp "$source" "$staging/$destination"
}

copy_file docs/ral/sne_roadseg_orfd/README.md README.md
copy_file docs/ral/sne_roadseg_orfd/source_provenance.json provenance/source_provenance.json
copy_file docs/ral/sne_roadseg_orfd/RESULTS.md results/RESULTS.md
copy_file docs/ral/EVALUATION_PROTOCOL.md docs/ral/EVALUATION_PROTOCOL.md
copy_file configs/environments/litevilnet_ral.yml configs/environments/litevilnet_ral.yml
copy_file requirements.txt requirements.txt
copy_file pytest.ini pytest.ini

for path in \
  tools/cache_official_orfd_normals.py \
  tools/train_matched_orfd_baseline.py \
  tools/evaluate_orfd.py \
  tools/summarize_matched_orfd_baselines.py \
  tools/write_sne_roadseg_report.py \
  tools/finalize_sne_roadseg_manuscript.py \
  tools/sanitize_table1_supplement.py \
  tools/package_ral_sne_roadseg.sh \
  tools/fetch_matched_baseline_sources.sh; do
  copy_file "$path" "$path"
done

for path in configs/splits/kitti_road/manifest_metadata.json; do
  copy_file "$path" "$path"
done

# Stage only the small JSON records.  The formal run directories also contain
# multi-gigabyte checkpoints, which must never enter an anonymous supplement.
for seed in 40 41 42; do
  mkdir -p "$raw/formal/sne_roadseg_seed${seed}"
  cp "$RESULT_ROOT/sne_roadseg_seed${seed}/result.json" \
    "$raw/formal/sne_roadseg_seed${seed}/result.json"
done
python tools/sanitize_table1_supplement.py \
  --source-results "$raw/formal" \
  --output-results "$raw/sanitized_formal"
python tools/sanitize_table1_supplement.py \
  --source-results "$SUMMARY_ROOT" \
  --output-results "$raw/docs/ral/orfd_matched_baselines/results"

for seed in 40 41 42; do
  copy_file "$raw/sanitized_formal/sne_roadseg_seed${seed}/result.json" \
    "results/seeds_sne_roadseg/sne_roadseg_orfd_seed${seed}.json"
done
for path in "$raw/docs/ral/orfd_matched_baselines/results/summary_sne_roadseg.json" \
            "$raw/docs/ral/orfd_matched_baselines/results/summary_sne_roadseg.csv"; do
  copy_file "$path" "results/$(basename "$path")"
done

scan_args=(--scan-root "$staging" --deny-token "$(id -un)" --deny-token "$(hostname)")
if [[ -n "${LITEVILNET_DOUBLE_BLIND_TOKENS:-}" ]]; then
  IFS=',' read -r -a private_tokens <<< "$LITEVILNET_DOUBLE_BLIND_TOKENS"
  for token in "${private_tokens[@]}"; do
    [[ -n "$token" ]] && scan_args+=(--deny-token "$token")
  done
fi
python tools/sanitize_table1_supplement.py "${scan_args[@]}"

mkdir -p "$(dirname "$OUTPUT")"
tar --sort=name --owner=0 --group=0 --numeric-owner \
  --mtime='UTC 2020-01-01 00:00:00' -cf - -C "$staging" . \
  | gzip -n > "$OUTPUT"
metadata_pattern="$(printf '%s' '/ho''me/|/Us''ers/|/ro''ot/')"
if tar -tvzf "$OUTPUT" | grep -Eiq "$metadata_pattern"; then
  echo "Anonymous archive metadata scan failed" >&2
  exit 1
fi
sha256sum "$OUTPUT" > "$OUTPUT.sha256"
echo "Created $OUTPUT"
cat "$OUTPUT.sha256"
