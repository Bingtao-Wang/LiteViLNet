# LiteViLNet Evaluation Protocol

## Metric Definitions

Local validation uses image-space binary segmentation masks from the repository split.

Reported fields:

- `MaxF`: maximum F1 after sweeping thresholds.
- `AP`: area under the precision-recall curve from the same threshold sweep.
- `PRE`: precision at the threshold that gives `MaxF`.
- `REC`: recall at the threshold that gives `MaxF`.
- `FPR`: false positive rate at the threshold that gives `MaxF`.
- `FNR`: false negative rate at the threshold that gives `MaxF`.
- `BestThreshold`: threshold that gives `MaxF`.

`F1@0.5` is a fixed-threshold debugging metric and is not used for main training selection or paper tables.

## Local Validation

`tools/train.py`, `tools/train_distill_edge.py`, `tools/train_ablation.py`, and `tools/evaluate.py` use `BinarySegmentationMeter` for threshold-swept metrics.

Best checkpoints are selected by validation `MaxF`.

Local validation is useful for iteration and ablation, but it is not identical to KITTI Road official evaluation.

## KITTI Road Official Results

Final paper-grade accuracy should come from the KITTI Road official server or a compatible devkit evaluation using exported prediction masks.

Export prediction masks with:

```bash
python -m tools.export_kitti_predictions \
  --preset litevilnet_baseline \
  --checkpoint weights/litevilnet/baseline/best_model.pth \
  --data_root data/kitti_road \
  --split test \
  --threshold 0.5
```

The default export location is:

```text
runs/kitti_submission/<preset>/
```

Binary masks are written directly under:

```text
runs/kitti_submission/<preset>/
```

The tool writes `manifest.json` with checkpoint, split, threshold, output count, and system metadata.

Until exported masks are evaluated by the official server or a compatible devkit, local validation results must be labeled as local image-space validation.

## Paper Tables

Use local validation for internal iteration tables and ablations. Use KITTI official/devkit results for final headline accuracy when available.

Deployment tables should pair the same checkpoint and input resolution with latency, FPS, memory, power, and energy measurements.
