# SNE-RoadSeg ORFD Formal Results

This anonymous report records the SNE-RoadSeg comparison on the official ORFD
held-out test partition. It is generated from the strict three-seed summary;
the source network and normal estimator are imported from the authors' official
repository at commit `5e7900bfd59887634ced687ffe85a73018a38659`.

## Protocol

- ORFD official train/validation/test partitions: 8,392/1,245/2,193 pairs.
- Network input: `704 x 1280`; fixed evaluator restores masks to original
  `1280 x 720` ground truth.
- 30 epochs, physical batch size 2, AMP, seeds 40/41/42.
- Checkpoint selection: validation 101-threshold MaxF.
- Fixed F/PRE/REC/IoU: argmax/0.5 mask, nearest-neighbor restoration, one
  foreground confusion matrix over all held-out pixels.
- AP and MaxF: complementary threshold sweeps on the common input grid.

## Seed Results

| Seed | Fixed F-score (%) | AP (%) | PRE (%) | REC (%) | IoU (%) |
|---:|---:|---:|---:|---:|---:|
| 40 | 94.3292 | 95.1572 | 92.5338 | 96.1958 | 89.2671 |
| 41 | 92.7953 | 98.4624 | 89.2783 | 96.6007 | 86.5589 |
| 42 | 93.5849 | 95.8077 | 91.0779 | 96.2338 | 87.9432 |
| Mean +/- sample SD | 93.5698+/-0.7671 | 96.4758+/-1.7509 | 90.9633+/-1.6307 | 96.3434+/-0.2236 | 87.9231+/-1.3542 |

## Reproduction

Create `litevilnet_ral` from `configs/environments/litevilnet_ral.yml`, obtain
ORFD from the official OFF-Net release, generate the official SNE normal cache,
and run `tools/train_matched_orfd_baseline.py` with `--baseline sne_roadseg`,
`--epochs 30`, `--batch-size 2`, `--height 704`, `--width 1280`, `--amp`, and
seeds 40, 41, and 42. Aggregate with
`tools/summarize_matched_orfd_baselines.py --methods sne_roadseg`; the strict
summarizer validates source hashes, split counts, checkpoint hashes, cache
metadata, and complete test-pixel coverage. The anonymous package is built by
`tools/package_ral_sne_roadseg.sh` and excludes data, checkpoints, caches, and
third-party source trees.
