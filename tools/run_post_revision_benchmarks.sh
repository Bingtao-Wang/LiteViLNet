#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
PYTHON="${LITEVILNET_PYTHON:-python}"
: "${LITEVILNET_KITTI_ROOT:?Set LITEVILNET_KITTI_ROOT to the KITTI Road data root}"
: "${LITEVILNET_KITTI_RUNS:?Set LITEVILNET_KITTI_RUNS to the KITTI ablation output root}"
: "${LITEVILNET_VELODYNE_ROOT:?Set LITEVILNET_VELODYNE_ROOT to extracted KITTI Velodyne data}"
: "${LITEVILNET_ORFD_ROOT:?Set LITEVILNET_ORFD_ROOT to the extracted ORFD Final_Dataset}"
: "${LITEVILNET_ROBOT_ROOT:?Set LITEVILNET_ROBOT_ROOT to the robot RGB-D data root}"
: "${LITEVILNET_ROBOT_SESSION:?Set LITEVILNET_ROBOT_SESSION to the session identifier}"
: "${LITEVILNET_ROBOT_CHECKPOINT:?Set LITEVILNET_ROBOT_CHECKPOINT to the robot checkpoint}"
KITTI_ROOT="${LITEVILNET_KITTI_ROOT}"
KITTI_RUNS="${LITEVILNET_KITTI_RUNS}"
VELODYNE_ROOT="${LITEVILNET_VELODYNE_ROOT}"
ORFD_ROOT="${LITEVILNET_ORFD_ROOT}"
ROBOT_ROOT="${LITEVILNET_ROBOT_ROOT}"
ROBOT_SESSION="${LITEVILNET_ROBOT_SESSION}"
ROBOT_CHECKPOINT="${LITEVILNET_ROBOT_CHECKPOINT}"

GPU_UUID="$(nvidia-smi -i "${GPU_ID}" --query-gpu=uuid --format=csv,noheader,nounits)"
ACTIVE_PROCESSES="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader,nounits | awk -F',' -v uuid="${GPU_UUID}" '$1 == uuid {print}')"
if [[ -n "${ACTIVE_PROCESSES}" ]]; then
  echo "Refusing a contaminated benchmark: GPU ${GPU_ID} (${GPU_UUID}) already has compute processes:" >&2
  echo "${ACTIVE_PROCESSES}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH=.
export NO_ALBUMENTATIONS_UPDATE=1

for repeat in 1 2 3; do
  "${PYTHON}" -m tools.profile_ablation \
    --img_h 384 --img_w 1248 --batch_size 1 --precision fp16 \
    --warmup 50 --iters 200 --skip_flops \
    --output "runs/revision_1/ablation_profile_4090d_clean_repeat_${repeat}.json"
done

"${PYTHON}" -m tools.summarize_profile_repeats \
  runs/revision_1/ablation_profile_4090d_clean_repeat_1.json \
  runs/revision_1/ablation_profile_4090d_clean_repeat_2.json \
  runs/revision_1/ablation_profile_4090d_clean_repeat_3.json \
  --cost-json runs/revision_1/ablation_cost_384x1248.json \
  --output runs/revision_1/ablation_profile_4090d_clean_summary.json

"${PYTHON}" -m tools.smoke_orfd_training \
  --data_root "${ORFD_ROOT}" --config full \
  --img_h 704 --img_w 1280 --batch_size 8 --num_workers 8 \
  --precision fp16 \
  --output runs/revision_1/orfd_full_batch8_training_smoke_complete.json

"${PYTHON}" -m tools.benchmark_kitti_adi_pipeline \
  --data_root "${KITTI_ROOT}" --velodyne_root "${VELODYNE_ROOT}" \
  --split_file configs/splits/kitti_road/stratified_seed20260723/val.txt \
  --checkpoint "${KITTI_RUNS}/full/seed_42/best_model.pth" \
  --img_h 384 --img_w 1248 --precision fp16 --warmup 5 --iters 58 \
  --output runs/revision_1/kitti_adi_end_to_end_4090d_clean.json

"${PYTHON}" -m tools.benchmark_robot_end_to_end \
  --data_root "${ROBOT_ROOT}" --session "${ROBOT_SESSION}" \
  --checkpoint "${ROBOT_CHECKPOINT}" \
  --img_h 800 --img_w 1280 --max_depth_mm 12000 \
  --precision fp16 --warmup 30 --iters 200 \
  --output runs/revision_1/robot_depth3_end_to_end_4090d_clean_800x1280.json
