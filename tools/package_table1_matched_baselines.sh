#!/usr/bin/env bash
set -euo pipefail

# Build the small, review-ready Table I reproducibility supplement.  Run from
# the LiteViLNet repository root after all twelve formal runs and five FPS
# measurements are summarized.

OUTPUT="${1:-dist/LiteViLNet_RAL_TableI_Reproduction.tar.gz}"
RESULT_ROOT="docs/ral/table1_matched_baselines/results"
SOURCE_RESULT_ROOT="${SOURCE_RESULT_ROOT:-${RESULT_ROOT}}"

required=(
  "${SOURCE_RESULT_ROOT}/summary.json"
  "${SOURCE_RESULT_ROOT}/summary.csv"
  "${SOURCE_RESULT_ROOT}/fps_4090d_summary.json"
  "${SOURCE_RESULT_ROOT}/fps_4090d_summary.csv"
  "docs/ral/table1_matched_baselines/README.md"
  "docs/ral/table1_matched_baselines/README_CN.md"
  "docs/ral/table1_matched_baselines/source_provenance.json"
)
for method in usnet sne_roadseg plard roadformer; do
  for seed in 40 41 42; do
    required+=("${SOURCE_RESULT_ROOT}/seeds/${method}_seed${seed}.json")
  done
done
for method in litevilnet usnet sne_roadseg plard roadformer; do
  required+=("${SOURCE_RESULT_ROOT}/fps/${method}.json")
done
for path in "${required[@]}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Required supplementary result is missing or empty: ${path}" >&2
    exit 1
  fi
done

files=(
  "pytest.ini"
  "configs/environments/litevilnet_ral.yml"
  "configs/environments/litevilnet_roadformer_ral.yml"
  "configs/splits/kitti_road/stratified_seed20260723/train.txt"
  "configs/splits/kitti_road/stratified_seed20260723/val.txt"
  "configs/splits/kitti_road/stratified_seed20260723/manifest_metadata.json"
  "docs/ral/table1_matched_baselines/README.md"
  "docs/ral/table1_matched_baselines/README_CN.md"
  "docs/ral/table1_matched_baselines/source_provenance.json"
  "litevilnet/__init__.py"
  "litevilnet/metrics/__init__.py"
  "litevilnet/metrics/deployment_metrics.py"
  "litevilnet/models/__init__.py"
  "litevilnet/models/attention_modules.py"
  "litevilnet/models/backbone.py"
  "litevilnet/models/decoder.py"
  "litevilnet/models/fusion_module.py"
  "litevilnet/models/litevilnet_rgbdepth.py"
  "litevilnet/models/losses.py"
  "litevilnet/models/vllinet.py"
  "litevilnet/models/vllinet_ablation.py"
  "tests/test_matched_kitti_baselines.py"
  "tools/cache_official_sne_normals.py"
  "tools/benchmark_matched_kitti_fps.py"
  "tools/fetch_matched_baseline_sources.sh"
  "tools/package_table1_matched_baselines.sh"
  "tools/prepare_matched_kitti_baselines.py"
  "tools/prepare_matched_roadformer.py"
  "tools/run_matched_kitti_baselines.sh"
  "tools/run_matched_kitti_fps.sh"
  "tools/sanitize_table1_supplement.py"
  "tools/summarize_matched_kitti_baselines.py"
  "tools/summarize_matched_kitti_fps.py"
  "tools/train_matched_kitti_baseline.py"
  "tools/train_matched_kitti_roadformer.py"
)

mkdir -p "$(dirname "${OUTPUT}")"
staging="$(mktemp -d)"
trap 'rm -rf "${staging}"' EXIT
for path in "${files[@]}"; do
  mkdir -p "${staging}/$(dirname "${path}")"
  cp "${path}" "${staging}/${path}"
done
python tools/sanitize_table1_supplement.py \
  --source-results "${SOURCE_RESULT_ROOT}" \
  --output-results "${staging}/${RESULT_ROOT}"
(
  cd "${staging}"
  find . -type f ! -name ARTIFACT_MANIFEST.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > ARTIFACT_MANIFEST.sha256
)
scan_args=(--scan-root "${staging}")
IFS=',' read -r -a deny_tokens <<< "${LITEVILNET_DOUBLE_BLIND_TOKENS:-}"
for token in "${deny_tokens[@]}"; do
  if [[ -n "${token}" ]]; then
    scan_args+=(--deny-token "${token}")
  fi
done
python tools/sanitize_table1_supplement.py "${scan_args[@]}"

# Normalize archive ownership and timestamps so neither the host username nor
# local UID/GID is encoded in tar metadata.  gzip -n omits its timestamp/name.
tar --sort=name --owner=0 --group=0 --numeric-owner \
  --mtime='UTC 2020-01-01 00:00:00' -cf - -C "${staging}" . \
  | gzip -n > "${OUTPUT}"
metadata_pattern="$(printf '%s' '/ho''me/|/Us''ers/')"
if tar -tvzf "${OUTPUT}" | grep -Eiq "${metadata_pattern}"; then
  echo "Anonymous archive metadata scan failed" >&2
  exit 1
fi
archive_sha256="$(sha256sum "${OUTPUT}" | cut -d' ' -f1)"
printf '%s  %s\n' "${archive_sha256}" "$(basename "${OUTPUT}")" > "${OUTPUT}.sha256"
echo "Created ${OUTPUT}"
cat "${OUTPUT}.sha256"
