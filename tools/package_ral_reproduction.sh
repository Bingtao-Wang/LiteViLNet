#!/usr/bin/env bash
set -euo pipefail

# Build the complete anonymous RA-L reproduction archive.  The source result
# roots contain only small result.json files; checkpoints are never copied.

OUTPUT="${1:-dist/LiteViLNet_RAL_Anonymous_Reproduction.tar.gz}"
: "${LITEVILNET_KITTI_RESULTS_ROOT:?Set LITEVILNET_KITTI_RESULTS_ROOT to the stratified KITTI ablation root}"
: "${LITEVILNET_KD_RESULTS_ROOT:?Set LITEVILNET_KD_RESULTS_ROOT to the KITTI KD result root}"
: "${LITEVILNET_ORFD_RESULTS_ROOT:?Set LITEVILNET_ORFD_RESULTS_ROOT to the ORFD ablation root}"

TABLE1_RESULTS="docs/ral/table1_matched_baselines/results"
required=(
  "${TABLE1_RESULTS}/summary.json"
  "${TABLE1_RESULTS}/summary.csv"
  "${TABLE1_RESULTS}/fps_4090d_summary.json"
  "${TABLE1_RESULTS}/fps_4090d_summary.csv"
)
for method in usnet sne_roadseg plard roadformer; do
  for seed in 40 41 42; do
    required+=("${TABLE1_RESULTS}/seeds/${method}_seed${seed}.json")
  done
done
for method in litevilnet usnet sne_roadseg plard roadformer; do
  required+=("${TABLE1_RESULTS}/fps/${method}.json")
done
for config in baseline add_lidar add_fusion add_bridge full optimal transformer_bridge; do
  for seed in 40 41 42; do
    required+=("${LITEVILNET_KITTI_RESULTS_ROOT}/${config}/seed_${seed}/result.json")
  done
done
for seed in 40 41 42; do
  required+=("${LITEVILNET_KD_RESULTS_ROOT}/seed_${seed}/result.json")
done
for config in full optimal; do
  for seed in 40 41 42; do
    required+=("${LITEVILNET_ORFD_RESULTS_ROOT}/${config}/seed_${seed}/result.json")
  done
done
for config in add_lidar add_fusion; do
  required+=("${LITEVILNET_ORFD_RESULTS_ROOT}/${config}/seed_42/result.json")
done
for path in "${required[@]}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Required reproduction evidence is missing or empty: ${path}" >&2
    exit 1
  fi
done

staging="$(mktemp -d)"
raw_evidence="$(mktemp -d)"
trap 'rm -rf "${staging}" "${raw_evidence}"' EXIT

copy_file() {
  local source="$1"
  local destination="${2:-${source}}"
  mkdir -p "${staging}/$(dirname "${destination}")"
  cp "${source}" "${staging}/${destination}"
}

copy_raw_evidence() {
  local source="$1"
  local destination="$2"
  mkdir -p "${raw_evidence}/$(dirname "${destination}")"
  cp "${source}" "${raw_evidence}/${destination}"
}

copy_file docs/ral/ANONYMOUS_SUPPLEMENT_README.md README.md
copy_file requirements.txt
copy_file pytest.ini

while IFS= read -r -d '' path; do
  copy_file "${path}"
done < <(find litevilnet tools tests -type f \( -name '*.py' -o -name '*.sh' \) -print0)
while IFS= read -r -d '' path; do
  copy_file "${path}"
done < <(find configs -type f \( -name '*.yml' -o -name '*.yaml' -o -name '*.json' -o -name '*.txt' \) -print0)
while IFS= read -r -d '' path; do
  copy_file "${path}"
done < <(find docs/ral/table1_matched_baselines -type f \( -name '*.md' -o -name '*.json' -o -name '*.csv' \) -print0)

for file in \
  kitti_stratified_summary.json \
  kitti_stratified_summary.csv \
  kitti_distillation_summary.json; do
  copy_raw_evidence "runs/revision_1/${file}" "kitti/${file}"
done
for file in \
  ablation_cost_384x1248.json \
  ablation_profile_4090d_clean_repeat_1.json \
  ablation_profile_4090d_clean_repeat_2.json \
  ablation_profile_4090d_clean_repeat_3.json \
  ablation_profile_4090d_clean_summary.json; do
  copy_raw_evidence "runs/revision_1/${file}" "profiling/${file}"
done
for file in \
  kitti_adi_end_to_end_4090d_clean.json \
  robot_depth3_end_to_end_4090d_clean_800x1280.json \
  orfd_full_batch8_training_smoke_complete.json; do
  copy_raw_evidence "runs/revision_1/${file}" "pipelines/${file}"
done
for file in orfd_summary.json orfd_summary.csv orfd_test_summary.json orfd_test_summary.csv; do
  copy_raw_evidence "runs/revision_1/${file}" "orfd/${file}"
done
for path in runs/revision_1/orfd_test/*.json; do
  copy_raw_evidence "${path}" "orfd/test_seeds/$(basename "${path}")"
done

for config in baseline add_lidar add_fusion add_bridge full optimal transformer_bridge; do
  for seed in 40 41 42; do
    copy_raw_evidence \
      "${LITEVILNET_KITTI_RESULTS_ROOT}/${config}/seed_${seed}/result.json" \
      "kitti/seed_results/${config}/seed_${seed}.json"
  done
done
for seed in 40 41 42; do
  copy_raw_evidence \
    "${LITEVILNET_KD_RESULTS_ROOT}/seed_${seed}/result.json" \
    "kitti/seed_results/kd_student/seed_${seed}.json"
done
for config in full optimal; do
  for seed in 40 41 42; do
    copy_raw_evidence \
      "${LITEVILNET_ORFD_RESULTS_ROOT}/${config}/seed_${seed}/result.json" \
      "orfd/seed_results/${config}/seed_${seed}.json"
  done
done
for config in add_lidar add_fusion; do
  copy_raw_evidence \
    "${LITEVILNET_ORFD_RESULTS_ROOT}/${config}/seed_42/result.json" \
    "orfd/seed_results/${config}/seed_42.json"
done

python tools/sanitize_table1_supplement.py \
  --source-results "${raw_evidence}" \
  --output-results "${staging}/evidence"
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

mkdir -p "$(dirname "${OUTPUT}")"
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
