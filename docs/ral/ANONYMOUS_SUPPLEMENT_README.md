# Anonymous RA-L Reproducibility Supplement for LiteViLNet

This archive accompanies the revised LiteViLNet manuscript.  It provides the
code, fixed split manifests, source provenance, seed-level numerical evidence,
and aggregation utilities needed to reproduce the reported KITTI Road and ORFD
experiments.  Datasets, pretrained checkpoints, generated normal caches, and
third-party repositories are intentionally excluded.  The current formal ORFD
comparison includes completed three-seed USNet and SNE-RoadSeg retraining under
the common held-out evaluator, together with a separate direct audit of the
authors' released OFF-Net checkpoint. The SNE-only entry point is
`tools/package_ral_sne_roadseg.sh`; incomplete local multi-seed runs are not
presented as finished evidence, and the paper-only OFF-Net published-test row
is omitted.

## 1. Contents and evidence map

| Manuscript evidence | Reproduction entry point | Archived numerical evidence |
|---|---|---|
| Table I matched KITTI accuracy and RTX 4090 D FPS-1 | `tools/run_matched_kitti_baselines.sh`, `tools/run_matched_kitti_offnet.sh`, `tools/run_matched_kitti_fps.sh`, `tools/dispatch_offnet_fps_after_idle.sh` | `docs/ral/table1_matched_baselines/results/` |
| Table II KITTI ablations | `tools/run_revision_ablation_queue.sh` | `evidence/kitti/` |
| Transformer and KD controls | `tools/run_revision_ablation_queue.sh`, `tools/run_kitti_distill_queue.sh` | `evidence/kitti/` |
| Table II ORFD evaluation | `tools/run_orfd_revision_queue.sh`, `tools/evaluate_orfd.py`, `tools/run_matched_orfd_baselines.sh`, `tools/dispatch_orfd_followup.sh` | `evidence/orfd/`, `docs/ral/orfd_matched_baselines/results/` |
| SNE-RoadSeg ORFD three-seed audit | `tools/train_matched_orfd_baseline.py`, `tools/summarize_matched_orfd_baselines.py`, `tools/package_ral_sne_roadseg.sh` | `docs/ral/orfd_matched_baselines/results/summary_sne_roadseg.{json,csv}` and `results/seeds_sne_roadseg/` |
| Parameters, MAC-equivalents, memory, and latency | `tools/profile_ablation.py`, `tools/summarize_profile_repeats.py` | `evidence/profiling/` |
| KITTI and RGB-D pipeline timing | `tools/benchmark_kitti_adi_pipeline.py`, `tools/benchmark_robot_end_to_end.py` | `evidence/pipelines/` |

Seed JSON files preserve metrics, hyperparameters, software/hardware versions,
and checkpoint hashes.  Local absolute paths are replaced by portable
placeholders only in the archived copies; the metrics and hashes are unchanged.

## 2. Environment

```bash
conda env create -f configs/environments/litevilnet_ral.yml
conda activate litevilnet_ral
```

The main recorded runs use Python 3.10, PyTorch 2.7.1 with CUDA 12.8, and
NVIDIA RTX 4090 D GPUs. LiteViLNet, USNet, and SNE-RoadSeg use automatic mixed
precision; PLARD and the legacy OpenMMLab graphs retain their documented FP32
paths. Device timing protocols are recorded separately in the manuscript and
evidence JSON.
RoadFormer uses the separately pinned
`configs/environments/litevilnet_roadformer_ral.yml` (Python 3.8, PyTorch
1.13.1+cu117, MMCV-full 1.7.0) required by its official source.

The supplied benchmark command records the same model-only protocol for every
architecture. If a target GPU is occupied at packaging time, an unmeasured
cell remains explicitly encoded as `--` rather than being inferred from a
different backend or device.

## 3. Data

- KITTI Road: obtain RGB images, road labels, calibration, and LiDAR data from
  the official benchmark.  The revision uses the versioned category-stratified
  231/58 manifests under
  `configs/splits/kitti_road/stratified_seed20260723/`.
- ORFD: obtain the released `Final_Dataset` and retain its official
  training/validation/testing partitions.
- Matched USNet/SNE-RoadSeg/PLARD/RoadFormer/OFF-Net inputs: follow
  `docs/ral/table1_matched_baselines/README.md`, including the official
  SNE-RoadSeg `depth_u16` archive and its recorded SHA-256.
- Matched ORFD baseline geometry preparation follows
  `docs/ral/orfd_matched_baselines/README.md`. Generated normal caches are not
  included in the archive and are reconstructed from pinned authors' code.

No validation or testing sample is used for training.  Validation selects the
checkpoint; the held-out ORFD testing partition is evaluated only afterward.

## 4. KITTI LiteViLNet training

Set portable roots and run the two GPU queues.  The full revision uses seeds
40, 41, and 42 for every reported configuration.

```bash
export LITEVILNET_DATA_ROOT=/path/to/kitti_road
export LITEVILNET_KITTI_TRAIN_SPLIT=configs/splits/kitti_road/stratified_seed20260723/train.txt
export LITEVILNET_KITTI_VAL_SPLIT=configs/splits/kitti_road/stratified_seed20260723/val.txt
export LITEVILNET_REVISION_OUTPUT=runs/revision_1/kitti_ablation

bash tools/run_revision_ablation_queue.sh 0 \
  baseline:40 baseline:41 baseline:42 \
  add_lidar:40 add_lidar:41 add_lidar:42 \
  optimal:40 optimal:41 optimal:42

bash tools/run_revision_ablation_queue.sh 1 \
  add_fusion:40 add_fusion:41 add_fusion:42 \
  add_bridge:40 add_bridge:41 add_bridge:42 \
  full:40 full:41 full:42 \
  transformer_bridge:40 transformer_bridge:41 transformer_bridge:42
```

Aggregate all runs and paired seed differences:

```bash
python -m tools.summarize_revision_experiments \
  runs/revision_1/kitti_ablation \
  --pair add_lidar:baseline \
  --pair add_fusion:add_lidar \
  --pair add_bridge:add_fusion \
  --pair full:add_bridge \
  --pair full:optimal \
  --pair full:transformer_bridge \
  --output runs/revision_1/kitti_stratified_summary.json \
  --csv runs/revision_1/kitti_stratified_summary.csv
```

For the KD control, point to the validation-selected full-model teacher:

```bash
export LITEVILNET_KITTI_ROOT=/path/to/kitti_road
export LITEVILNET_KITTI_TEACHER=/path/to/full/seed_42/best_model.pth
export LITEVILNET_KITTI_DISTILL_OUTPUT=runs/revision_1/kitti_distill_edge
bash tools/run_kitti_distill_queue.sh 0 40 41 42
```

## 5. Matched-protocol Table I baselines

The complete official-source provenance, clone commands, data preparation,
training commands, evaluation definition, and aggregation rules are documented
in:

```text
docs/ral/table1_matched_baselines/README.md
docs/ral/table1_matched_baselines/source_provenance.json
```

No baseline network is reimplemented.  Model/loss definitions come directly
from the authors' official repositories at pinned commits.  The local adapter
only provides the common split, 150-epoch budget, seed control, method-specific
official optimization recipes, common evaluator, and provenance output.
The five baselines are USNet, SNE-RoadSeg, PLARD, RoadFormer, and OFF-Net, all fetched
from their authors' official GitHub repositories at the full commits recorded
in `source_provenance.json`. FPS-1 additionally uses one common RTX 4090 D,
`384 x 1248`, batch-1, PyTorch FP32, model-only CUDA-event protocol for all
six architectures.

## 6. ORFD training and held-out testing

```bash
export LITEVILNET_ORFD_ROOT=/path/to/Final_Dataset
export LITEVILNET_ORFD_OUTPUT=runs/revision_1/orfd_ablation

bash tools/run_orfd_revision_queue.sh 0 full:40 full:41 full:42
bash tools/run_orfd_revision_queue.sh 1 optimal:40 optimal:41 optimal:42
```

Evaluate each validation-selected checkpoint on the official testing
partition, then aggregate with `tools/summarize_orfd_test.py`.  The evaluator
uses an OFF-Net-style fixed argmax/0.5 convention, restores predictions to
the original ground-truth size, and accumulates one foreground confusion
matrix over all 2,193 testing frames.

For the locally reproduced ORFD comparison, follow
`docs/ral/orfd_matched_baselines/README.md`. The current formal evidence
contains three independently initialized USNet seeds under the common
held-out testing and fixed-argmax evaluator. The separate released OFF-Net
checkpoint audit is retained as a single direct measurement; the paper-only
published-test row is not included in the local comparison.

## 7. Verification

```bash
python -m pytest -q
python -m py_compile tools/*.py
```

The Table I summarizers additionally refuse mismatched seeds, split hashes,
input sizes, epoch budgets, method-specific AMP settings, official
remotes/commits, source-file hashes, parameter counts, checkpoint SHA-256
values, or FPS device/precision/timing scopes.

## 8. Double-blind packaging

This archive is generated by `tools/package_ral_reproduction.sh`.  Before
archiving, it replaces local paths in temporary evidence copies, scans file
names and contents for home/data paths, email addresses, SSH remotes, and
the runtime username/hostname automatically. Additional private identity tokens
are supplied through `LITEVILNET_DOUBLE_BLIND_TOKENS`. The packager normalizes
tar owner/group
to numeric `0/0`.  The original evidence is never rewritten.  The archive does
not contain the project homepage, manuscript source, images, `.git` history,
datasets, checkpoints, caches, or third-party source trees.
An internal `ARTIFACT_MANIFEST.sha256` records every archived payload with a
repository-relative path, in addition to the companion hash of the complete
compressed archive.

The archive is built after all formal experiments with:

```bash
export LITEVILNET_KITTI_RESULTS_ROOT=/path/to/kitti_ablation
export LITEVILNET_KD_RESULTS_ROOT=/path/to/kitti_distill_edge
export LITEVILNET_ORFD_RESULTS_ROOT=/path/to/orfd_ablation
export LITEVILNET_DOUBLE_BLIND_TOKENS="comma-separated private identity tokens"
bash tools/package_ral_reproduction.sh
```
