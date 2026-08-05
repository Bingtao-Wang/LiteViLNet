#!/usr/bin/env bash
set -euo pipefail

# Fetch only the authors' official repositories and pin the exact revisions
# used by the matched-protocol Table I experiments.  Third-party source is not
# copied into the LiteViLNet supplementary package.

TARGET_ROOT="${1:-third_party/matched_baselines}"

clone_or_pin() {
  local name="$1"
  local repository="$2"
  local commit="$3"
  local target="${TARGET_ROOT}/${name}"

  if [[ ! -e "${target}" ]]; then
    git clone "${repository}" "${target}"
  elif [[ ! -d "${target}/.git" ]]; then
    echo "Refusing to use non-git path: ${target}" >&2
    return 1
  fi

  local origin
  origin="$(git -C "${target}" remote get-url origin)"
  if [[ "${origin}" != "${repository}" ]]; then
    echo "Refusing non-official origin for ${name}: ${origin}" >&2
    return 1
  fi

  if [[ -n "$(git -C "${target}" status --porcelain)" ]]; then
    echo "Refusing to change a non-clean source tree: ${target}" >&2
    return 1
  fi

  git -C "${target}" fetch origin "${commit}"
  git -C "${target}" checkout --detach "${commit}"

  local actual
  actual="$(git -C "${target}" rev-parse HEAD)"
  if [[ "${actual}" != "${commit}" ]]; then
    echo "Commit verification failed for ${name}: ${actual}" >&2
    return 1
  fi
  printf '%s\t%s\t%s\n' "${name}" "${repository}" "${actual}"
}

mkdir -p "${TARGET_ROOT}"
clone_or_pin \
  USNet \
  https://github.com/morancyc/USNet.git \
  d761158ad42df7dcb62fa257dd02ce11c85f94a5
clone_or_pin \
  SNE-RoadSeg \
  https://github.com/hlwang1124/SNE-RoadSeg.git \
  5e7900bfd59887634ced687ffe85a73018a38659
clone_or_pin \
  PLARD \
  https://github.com/zhechen/PLARD.git \
  44485803092e729661c696ab6c03f6f2fabc8701
clone_or_pin \
  Road-Former \
  https://github.com/LiJiahang617/Road-Former.git \
  f675a3467cb168ebc727648390c304279bbcb079
clone_or_pin \
  OFF-Net \
  https://github.com/chaytonmin/Off-Road-Freespace-Detection \
  50e63d24836198e8fb5af707e521f414104b4876

echo "Pinned official sources are ready under ${TARGET_ROOT}."
