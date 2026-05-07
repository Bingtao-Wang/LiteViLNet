# VLLiNet to LiteViLNet Migration

## Source

- Requested source path: `/home/admin1/Mycode/Master_Thesis/Ch2_VLLiNet`
- Real source path: `/home/admin1/Mycode/VLLiNet`
- New workspace: `/home/admin1/Mycode/LiteViLNet`

## Copied

- Core model code from `models/` to `litevilnet/models/`.
- KITTI dataset and metric helpers from `utils/` to `litevilnet/data/` and `litevilnet/metrics/`.
- Deployment and benchmark entry points from `deployment/` to `tools/`.
- Seed checkpoints:
  - `vllinet_paper_v3_final.pth`
  - `vllinet_edge_add_lidar.pth`

## Linked

- `datasets/data` links to `/home/admin1/Mycode/VLLiNet/datasets/data`.

## Not Copied

- Historical `experiments/`, `logs/`, `releases/`, figure folders, CARLA integrations, and old variants.
- VLLiNet V6/Acc checkpoint is intentionally not part of the LiteViLNet mainline.

## Compatibility

Core CLI arguments are preserved where practical: `--data_root`, `--checkpoint`, `--preset`, `--precision`, and `--output`.
Default paths now point to the LiteViLNet workspace.
