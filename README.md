# LiteViLNet

LiteViLNet is a clean research workspace derived from VLLiNet for efficient RGB-LiDAR road segmentation on KITTI Road and embedded platforms.

The repository is intentionally small:

- Core code lives under `litevilnet/`.
- Entry points live under `tools/`.
- KITTI/CARLA data is linked from the original VLLiNet data store via `datasets/data`.
- Seed checkpoints are stored locally but ignored by Git.
- External KITTI leaderboard models must stay in `third_party/` and connect through `adapters/`.

## Quick Checks

```bash
python -m tools.evaluate \
  --preset vllinet_paper \
  --checkpoint checkpoints/seed_from_vllinet/vllinet_paper_v3_final.pth \
  --data_root datasets/data/kitti_road \
  --split val
```

```bash
python -m tools.benchmark_pytorch \
  --preset vllinet_edge \
  --checkpoint checkpoints/seed_from_vllinet/vllinet_edge_add_lidar.pth \
  --precision fp16
```

```bash
python -m tools.export_onnx \
  --preset litevillinet_baseline \
  --output deployment/artifacts/onnx/litevillinet_baseline_384x1248.onnx
```

## Presets

- `vllinet_paper`: 14.04M parameter accuracy reference seeded from VLLiNet V3 Final.
- `vllinet_edge`: 3.43M parameter lightweight reference seeded from the VLLiNet add-LiDAR ablation.
- `litevillinet_baseline`: first LiteViLNet iteration baseline, initialized from the lightweight architecture.
