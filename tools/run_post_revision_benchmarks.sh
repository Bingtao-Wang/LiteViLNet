#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
PYTHON="/home/aihub/miniconda3/envs/litevilnet_ral/bin/python"
KITTI_ROOT="/data/Database/Research04-LiteViLNet/LiteViLNet/data/kitti_road"
KITTI_RUNS="/data/Database/Research04-LiteViLNet/revision_1_runs/kitti_ablation_stratified_seed20260723"
VELODYNE_ROOT="/data/Database/Research04-LiteViLNet/datasets/KITTI_Road/velodyne_extracted"
ORFD_ROOT="${LITEVILNET_ORFD_ROOT:-/data/Database/Research04-LiteViLNet/datasets/ORFD/extracted/Final_Dataset}"
ROBOT_ROOT="/data/Database/Research04-LiteViLNet/LiteViLNet/data/robot_road_raw"
ROBOT_SESSION="wheeled_campus_road_20260511_142931"
ROBOT_CHECKPOINT="/data/Database/Research04-LiteViLNet/LiteViLNet/output/3.1_litevilnet_rgbdepth_robot_path_wheeled_campus_road_20260511_142931_labeled20_sampled20_smoke/best_model.pth"

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
