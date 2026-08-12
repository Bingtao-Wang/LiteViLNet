# Reproducing the Matched-Protocol KITTI Table I

This package reproduces the local accuracy rows used in Table I of the revised
LiteViLNet manuscript.  Its purpose is to compare LiteViLNet and five public
RGB--geometry baselines under one executable protocol: the same 231/58
category-stratified KITTI Road split, `384 x 1248` network input, and the same
pixel-accumulated 101-threshold perspective-view MaxF evaluator.

## 1. Official sources

No baseline architecture is reimplemented in the LiteViLNet repository.
The adapter imports the model and loss definitions from the following official
repositories.

| Method | Official repository | Pinned commit | Imported definitions |
|---|---|---|---|
| USNet | <https://github.com/morancyc/USNet> | `d761158ad42df7dcb62fa257dd02ce11c85f94a5` | `model/usnet.py`, `loss.py` |
| SNE-RoadSeg | <https://github.com/hlwang1124/SNE-RoadSeg> | `5e7900bfd59887634ced687ffe85a73018a38659` | `models/networks.py`, `models/roadseg_model.py`, `models/sne_model.py` |
| PLARD | <https://github.com/zhechen/PLARD> | `44485803092e729661c696ab6c03f6f2fabc8701` | `ptsemseg/models/plard.py`, `ptsemseg/loss.py`, `ptsemseg/loader/kitti_road_loader.py` |
| RoadFormer | <https://github.com/LiJiahang617/Road-Former> | `f675a3467cb168ebc727648390c304279bbcb079` | official TwinConvNeXt backbone, RoadFormer head/pixel decoder, and KITTI config |
| OFF-Net | <https://github.com/chaytonmin/Off-Road-Freespace-Detection> | `50e63d24836198e8fb5af707e521f414104b4876` | `models/transformer_models`, `models/loss.py`, `models/sne_model.py` |

The direct table uses a reproducibility-based inclusion rule: an authors'
official source tree must expose the complete trainable graph and required
input path, and we must be able to retrain and time that exact graph locally.
Methods for which only paper/server values or an incompatible third-party
legacy port could be verified remain in Related Work but are not numerically
ranked against local perspective-view runs. This rule yields three CNN fusion
baselines (USNet/SNE-RoadSeg/OFF-Net), the classic RGB--LiDAR PLARD
architecture, and the Transformer-based RGB--normal RoadFormer architecture.

The official files used on our machine differed from the pinned Git objects
only in CRLF/LF line endings (`git diff --ignore-space-at-eol --exit-code`
against `HEAD` returns zero).  Every seed-level `result.json` records the repository remote,
commit, imported-file SHA-256 values, checkpoint SHA-256, command line, package
versions, and GPU model.  The training adapter refuses to start when that
semantic-diff check fails unless an explicit diagnostic-only override is used.
The same pins and content hashes are available in
`source_provenance.json` for machine-readable auditing.
Each tracked source records its Git blob, canonical fresh-clone LF SHA-256, and
the CRLF working-copy SHA-256 used for the formal run, so either checkout style
can be verified without treating line endings as a code change.

The networks and losses are imported directly.  The split-safe local data and
training loops mirror the recipes documented by the pinned USNet `train.py`,
`utils.py`, `dataset/kitti.py`, and `dataset/custom_transforms.py`; SNE-RoadSeg
`train.py`, `data/kitti_dataset.py`, `options/*.py`, `models/networks.py`, and
`models/roadseg_model.py`; PLARD `train.py`, loader, model, and loss; OFF-Net
`train.py`, ORFD loader, model, loss, and SNE; and the RoadFormer KITTI
configuration plus its official custom OpenMMLab modules. Hashes for these recipe
sources are included in `source_provenance.json`; this makes the small adapter
boundary inspectable rather than presenting a reimplementation as official
code.

Create the pinned source trees with:

```bash
bash tools/fetch_matched_baseline_sources.sh third_party/matched_baselines
export USNET_SOURCE="$PWD/third_party/matched_baselines/USNet"
export SNE_SOURCE="$PWD/third_party/matched_baselines/SNE-RoadSeg"
export PLARD_SOURCE="$PWD/third_party/matched_baselines/PLARD"
export ROADFORMER_SOURCE="$PWD/third_party/matched_baselines/Road-Former"
export OFFNET_SOURCE="$PWD/third_party/matched_baselines/OFF-Net"
```

The fetch script uses the authors' official GitHub repositories, not mirrors
or forks.  It refuses to overwrite a non-Git or dirty target and checks out the
exact detached commits above.  The training adapter independently verifies the
commit and semantic work-tree diff before a run starts.

USNet uses the official ImageNet ResNet-18 weights distributed by torchvision:

```text
f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec
```

SNE-RoadSeg uses the official source's ImageNet ResNet-152 initialization:

```text
394f9c45966e3651a89bbb78a48410a6755854ce4a5ab64927cf1c7247f85e58
```

RoadFormer uses the ConvNeXt-Base ImageNet-21K checkpoint URL embedded in its
official KITTI config; the downloaded file SHA-256 is:

```text
262fd0376855955f20f6c036aa882f5cb22b88333b766b0fa20174339c11d70d
```

## 2. Data provenance

Download the RGB images, labels, and calibration files from the KITTI Road
benchmark.  PLARD ADIs used by LiteViLNet are the released PLARD training ADIs.
For USNet and SNE-RoadSeg, use the `depth_u16` archive linked by the official
SNE-RoadSeg README:

<https://drive.google.com/file/d/1phoi_f3bwEV-oKwGe0psXj5XDhqy5DH0/view>

USNet reads this depth to run its vendored official SNE and feeds the resulting
three-channel surface normal to the network. Its Table-I network input is
therefore reported as `RGB+Normal`, rather than inferred from the internal
`depth` variable name.

Archive SHA-256:

```text
d32bf0052ec81f87996c0c7ca2e86952b9f780ad97b7347151e62def5f8efb92
```

It can be fetched and verified with:

```bash
gdown 1phoi_f3bwEV-oKwGe0psXj5XDhqy5DH0 -O depth_u16.zip
sha256sum depth_u16.zip
unzip depth_u16.zip -d /path/to/extracted_depth_archive
```

The split manifests are versioned at:

```text
configs/splits/kitti_road/stratified_seed20260723/train.txt
configs/splits/kitti_road/stratified_seed20260723/val.txt
```

Their SHA-256 values are:

```text
train  93a8b849a531e9bd938c65120816f5ad4bd62f563e7f0d68ac6c0e6046425867
val    69b10e5ff641d5cea81d2f0832ada2c31ee5f3b3f8ced9e4e962f889a722976f
```

`tools/prepare_matched_kitti_baselines.py` checks every required RGB, label,
calibration, ADI, and depth file, rejects duplicate IDs or train/validation
overlap, and creates a symlink-only baseline tree.  USNet calls its validation
directory `validating`, whereas SNE-RoadSeg calls it `validation`; both names
point to the same 58 samples.

## 3. Environment

```bash
conda env create -f configs/environments/litevilnet_ral.yml
conda activate litevilnet_ral
```

USNet, SNE-RoadSeg, PLARD, and LiteViLNet use this environment. The recorded
runs use Python 3.10.20, PyTorch 2.7.1+cu128, torchvision
0.22.1+cu128, NumPy 1.26.4, OpenCV 4.10.0, and two RTX 4090 D 48-GB GPUs.
The adapter retains the deprecated `pretrained=True` calls in the official
source; torchvision maps these calls to the cited ImageNet V1 weights.

RoadFormer and OFF-Net are run in the older pinned environment used by their
official source dependencies:

```bash
conda env create -f configs/environments/litevilnet_roadformer_ral.yml
```

The verified RoadFormer environment uses Python 3.8.20, PyTorch 1.13.1+cu117,
torchvision 0.14.1+cu117, and MMCV-full 1.7.0. The adapter imports the official
Python graph without editing it and maps its vendored `mmcv_custom` operator
names to the matching installed MMCV 1.7 CUDA operators. The pinned
multi-scale deformable-attention operator is FP32-only, so RoadFormer training
and the common Table-I FPS measurement use FP32.

## 4. Prepare inputs

The following example assumes that the SNE-RoadSeg depth archive was extracted
as `$DEPTH_ROOT/depth_u16/{training,testing}`.

```bash
export KITTI_ROOT=/path/to/kitti_road
export DEPTH_ROOT=/path/to/extracted_depth_archive
export MATCHED_ROOT=runs/revision_1/matched_baselines/kitti_road

python tools/prepare_matched_kitti_baselines.py \
  --data-root "$KITTI_ROOT" \
  --depth-root "$DEPTH_ROOT/depth_u16" \
  --train-file configs/splits/kitti_road/stratified_seed20260723/train.txt \
  --val-file configs/splits/kitti_road/stratified_seed20260723/val.txt \
  --output-root "$MATCHED_ROOT"
```

The official SNE calculation is deterministic but slow when repeated inside
every epoch.  Cache its exact float32 output once.  Both the original and the
official USNet pre-SNE horizontal-flip variants are retained.

USNet also vendors `model/sne_model.py`.  Relative to the SNE-RoadSeg authors'
file, it only exposes the already-active top-crop branch as a default-true
`crop_top` option; USNet's official loader uses that default.  Consequently,
the pinned implementations execute the same numerical path here.  Both
official file hashes are recorded in `source_provenance.json`.

```bash
python tools/cache_official_sne_normals.py \
  --data-root "$MATCHED_ROOT" \
  --official-source /path/to/SNE-RoadSeg \
  --output-root "$MATCHED_ROOT" \
  --workers 8
```

The cache metadata records the official SNE file hash and commit.  Caching is
only a deterministic preprocessing optimization; it does not change the
normal estimator or network inputs.

OFF-Net uses its own pinned SNE implementation, so its cache is intentionally
separate from the SNE-RoadSeg/USNet cache:

```bash
export KITTI_OFFNET_NORMAL_ROOT=runs/revision_1/matched_baselines/kitti_offnet_normals
conda run -n litevilnet_roadformer_ral env PYTHONPATH=. \
  python tools/cache_official_sne_normals.py \
  --data-root "$MATCHED_ROOT" --official-source "$OFFNET_SOURCE" \
  --output-root "$KITTI_OFFNET_NORMAL_ROOT" --profile offnet --workers 4
```

RoadFormer's official loader expects the three-channel normal field as a PNG.
The preparation adapter deterministically maps the cached float32 normal from
`[-1,1]` to uint16, creates a separate split-safe tree, and records the same
manifest hashes:

```bash
export ROADFORMER_DATA_ROOT=runs/revision_1/matched_baselines/roadformer_kitti
python tools/prepare_matched_roadformer.py \
  --matched-root "$MATCHED_ROOT" \
  --train-file configs/splits/kitti_road/stratified_seed20260723/train.txt \
  --val-file configs/splits/kitti_road/stratified_seed20260723/val.txt \
  --output-root "$ROADFORMER_DATA_ROOT"
```

## 5. Train one seed

USNet retains its official symmetric ResNet-18, evidential loss with a
50-epoch KL annealing horizon, AdamW optimizer, backbone learning-rate ratio,
and polynomial schedule.  Its official geometric/RGB augmentations are kept.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_matched_kitti_baseline.py \
  --baseline usnet \
  --official-source /path/to/USNet \
  --data-root "$MATCHED_ROOT" \
  --output-dir runs/revision_1/matched_baselines/formal/usnet_seed40 \
  --seed 40 --epochs 150 --batch-size 2 --num-workers 4 \
  --val-every 5 --early-stop-validations 20 --amp --device cuda
```

SNE-RoadSeg retains the official dual ResNet-152 source model, ImageNet
initialization, decoder initialization, cross-entropy objective, SGD optimizer,
and stepwise decay recipe.

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train_matched_kitti_baseline.py \
  --baseline sne_roadseg \
  --official-source /path/to/SNE-RoadSeg \
  --data-root "$MATCHED_ROOT" \
  --output-dir runs/revision_1/matched_baselines/formal/sne_roadseg_seed40 \
  --seed 40 --epochs 150 --batch-size 2 --num-workers 4 \
  --val-every 5 --early-stop-validations 20 --amp --device cuda
```

PLARD retains the official RGB--ADI graph, three supervised outputs, SGD
optimizer, frozen batch normalization, and the gradual learning-rate decay
specified in its paper. The official KITTI checkpoint is not used to initialize
these runs because it was trained on all 289 KITTI training images and would
therefore leak the local validation partition.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_matched_kitti_baseline.py \
  --baseline plard --official-source "$PLARD_SOURCE" \
  --data-root "$MATCHED_ROOT" \
  --output-dir runs/revision_1/matched_baselines/formal/plard_seed40 \
  --seed 40 --epochs 150 --batch-size 4 --num-workers 4 \
  --val-every 5 --early-stop-validations 20 --device cuda
```

RoadFormer retains the official TwinConvNeXt-B backbone, ImageNet-21K
initialization, RoadFormer pixel/transformer decoder, Hungarian matching
losses, AdamW parameter groups, gradient clipping, and polynomial decay.
Batch 4 is used on one 48-GB GPU; its official MMCV deformable-attention CUDA
operator is run in its supported FP32 path.

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output \
  -n litevilnet_roadformer_ral env PYTHONPATH=. \
  python tools/train_matched_kitti_roadformer.py \
  --official-source "$ROADFORMER_SOURCE" \
  --data-root "$ROADFORMER_DATA_ROOT" \
  --output-dir runs/revision_1/matched_baselines/formal/roadformer_seed40 \
  --seed 40 --epochs 150 --batch-size 4 --num-workers 4 \
  --val-every 5 --early-stop-validations 20
```

OFF-Net retains its official MiT-B2 RGB--normal graph, Kaiming
initialization, cross-entropy loss, SGD optimizer, and stepwise decay. Its
native 1/4-resolution training target is constructed directly from the KITTI
ground truth.

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output \
  -n litevilnet_roadformer_ral env PYTHONPATH=. \
  python tools/train_matched_kitti_offnet.py \
  --official-source "$OFFNET_SOURCE" --data-root "$MATCHED_ROOT" \
  --normal-root "$KITTI_OFFNET_NORMAL_ROOT" \
  --output-dir runs/revision_1/matched_baselines/formal/offnet_seed40 \
  --seed 40 --epochs 150 --batch-size 2 --num-workers 4 \
  --height 384 --width 1248 --val-every 5 --early-stop-validations 20
```

Repeat all five commands for seeds 41 and 42.  The common 150-epoch budget matches
the LiteViLNet runs.  Best checkpoints are selected by validation MaxF for all
methods.

## 6. Evaluation and aggregation

The runner accumulates all validation pixels before scanning thresholds
`0.00, 0.01, ..., 1.00`.  PRE, REC, FPR, FNR, and IoU are taken at the MaxF
threshold.  This is the same `BinarySegmentationMeter` used for the LiteViLNet
rows; no method-specific fixed threshold is used in Table I.

```bash
python tools/summarize_matched_kitti_baselines.py \
  --input-root runs/revision_1/matched_baselines/formal \
  --expected-seeds 40,41,42 \
  --output-json docs/ral/table1_matched_baselines/results/summary.json \
  --output-csv docs/ral/table1_matched_baselines/results/summary.csv \
  --seed-output-dir docs/ral/table1_matched_baselines/results/seeds \
  --anonymous-seed-copies
```

The summarizer refuses to combine runs if seeds, input dimensions, sample
counts, epoch budget, AMP setting, official commits, parameter counts, or split
hashes differ.  Reported uncertainty is the sample standard deviation over the
three independent runs.

The flag anonymizes only the seed JSON copies placed under `docs/`; the raw
run JSON under `runs/` remains unchanged for local audit.  Metrics, source and
checkpoint hashes, commits, hardware, and software versions are preserved.

### Recorded formal results

The completed formal runs produce the following perspective-view validation
statistics (percent; mean $\pm$ sample standard deviation):

| Method | Seeds | MaxF | PRE | REC | Parameters |
|---|---|---:|---:|---:|---:|
| USNet | 40/41/42 | 97.88 $\pm$ 0.07 | 98.03 $\pm$ 0.03 | 97.73 $\pm$ 0.11 | 30.74M |
| SNE-RoadSeg | 40/41/42 | 97.23 $\pm$ 0.21 | 97.39 $\pm$ 0.30 | 97.06 $\pm$ 0.13 | 201.32M |
| PLARD | 40/41/42 | 95.25 $\pm$ 0.19 | 95.46 $\pm$ 0.29 | 95.03 $\pm$ 0.09 | 76.93M |
| OFF-Net | 40/41/42 | 95.36 $\pm$ 0.66 | 94.88 $\pm$ 0.92 | 95.83 $\pm$ 0.57 | 25.21M |
| RoadFormer | 40/41/42 | 97.28 $\pm$ 0.05 | 97.96 $\pm$ 0.09 | 96.61 $\pm$ 0.16 | 206.86M |
| LiteViLNet | 40/41/42 | 97.23 $\pm$ 0.15 | 97.31 $\pm$ 0.59 | 97.16 $\pm$ 0.30 | 14.04M |

The exact unrounded values, best epochs, source hashes, and all baseline checkpoint
SHA-256 values are in `results/summary.json` and `results/seeds/*.json`.  The
LiteViLNet's row is generated by the main revision queues documented in the
complete anonymous supplement.

The matched RTX 4090 D FPS-1 snapshot is:

| Method | Parameters | FPS-1 |
|---|---:|---:|
| USNet | 30.74M | 239.81 |
| SNE-RoadSeg | 201.32M | 19.32 |
| PLARD | 76.93M | 26.97 |
| RoadFormer | 206.86M | 17.46 |
| OFF-Net | 25.21M | 65.04 |
| LiteViLNet | 14.04M | 216.61 |

FPS-1 uses RTX 4090 D, PyTorch FP32, batch 1, `384 x 1248`, resident inputs,
100 warmups, 300 timed iterations, and three independent repeats; it measures
only the model forward. Jetson FPS-2 is reported in the manuscript only for
the configurations with a matching Orin NX measurement.

## 7. Matched RTX 4090 D FPS-1

FPS-1 is remeasured locally for all six exact architectures. Each command uses
the best checkpoint from one completed matched seed, a resident random RGB--geometry
pair, `384 x 1248`, batch 1, PyTorch FP32, 100 warmups, 300 timed iterations,
and three repeats. CUDA events bracket only the model forward; decoding,
preprocessing, host-to-device transfer, and postprocessing are excluded.

Use `tools/run_matched_kitti_fps.sh` to run all six measurements sequentially
on one otherwise idle target GPU. USNet, SNE-RoadSeg, PLARD, and LiteViLNet run
in the active `litevilnet_ral` environment; the wrapper launches RoadFormer
and OFF-Net in `litevilnet_roadformer_ral` automatically. Point each variable to one
validation-selected checkpoint from the completed matched runs:

```bash
export LITEVILNET_CHECKPOINT=/path/to/litevilnet/full/best_model.pth
export USNET_CHECKPOINT=runs/revision_1/matched_baselines/formal/usnet_seed40/best_model.pth
export SNE_CHECKPOINT=runs/revision_1/matched_baselines/formal/sne_roadseg_seed40/best_model.pth
export PLARD_CHECKPOINT=runs/revision_1/matched_baselines/formal/plard_seed40/best_model.pth
export ROADFORMER_CHECKPOINT=runs/revision_1/matched_baselines/formal/roadformer_seed40/best_model.pth
export OFFNET_CHECKPOINT=runs/revision_1/matched_baselines/formal/offnet_seed40/best_model.pth
export FPS_GPU=0
bash tools/run_matched_kitti_fps.sh
```

The wrapper calls `benchmark_matched_kitti_fps.py` once per method and then
validates and aggregates `litevilnet.json`, `usnet.json`, `sne_roadseg.json`,
`plard.json`, `roadformer.json`, and `offnet.json`. To rerun only the aggregation step:

```bash
python tools/summarize_matched_kitti_fps.py \
  --input-root runs/revision_1/matched_baselines/fps_4090d \
  --output-json docs/ral/table1_matched_baselines/results/fps_4090d_summary.json \
  --output-csv docs/ral/table1_matched_baselines/results/fps_4090d_summary.csv \
  --result-output-dir docs/ral/table1_matched_baselines/results/fps \
  --anonymous-result-copies
```

The summarizer rejects a device, precision, input shape, timing scope,
iteration-count, parameter-count, or source-commit mismatch.

## 8. Scope of local adapters

The LiteViLNet-side code performs only:

1. split-safe file discovery and symlink creation;
2. deterministic caching of the official SNE output;
3. a compact training loop that calls the official model/loss and mirrors each
   official optimizer, schedule, initialization, and augmentation recipe while
   adding command-line seeds and output metadata;
4. the common threshold-swept evaluator and multi-seed aggregation.

It does not replace the baseline encoders, fusion blocks, decoders, losses, or
pretrained initialization.  The local results are explicitly labeled as
perspective-view validation and are not mixed with KITTI official BEV
test-server scores.

## 9. Build the review supplement

After all fifteen baseline seed JSON files and six FPS JSON files have been generated, build a self-contained code
and evidence archive with:

```bash
bash tools/package_table1_matched_baselines.sh
```

This produces `dist/LiteViLNet_RAL_TableI_Reproduction.tar.gz` and a companion
SHA-256 file.  The packager refuses to run if any seed result or summary is
missing.  It includes the adapters, evaluator, tests, environment, manifests,
manuals, provenance, and seed/aggregated JSON, but intentionally excludes KITTI
data, normal caches, checkpoints, third-party source trees, and the project
homepage README.  Before archiving, a temporary copy replaces local absolute
paths with portable placeholders and scans for home paths, local account names,
email addresses, SSH remotes, the current runtime username/hostname, and any
comma-separated tokens supplied through `LITEVILNET_DOUBLE_BLIND_TOKENS`; the
host JSON is never rewritten. Tar
owner/group are normalized to numeric `0/0`, and gzip name/time metadata are
disabled.  Any detected double-blind leak aborts packaging.  The pinned official
sources are reconstructed by the fetch script.
The archive also contains `ARTIFACT_MANIFEST.sha256`, which records every
payload file using only repository-relative paths.
