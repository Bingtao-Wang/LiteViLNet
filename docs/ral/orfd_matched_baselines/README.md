# Reproducing the ORFD Baseline Evaluation

This supplement documents the applicable RGB--geometry methods on the
released ORFD training, validation, and testing partitions. The manuscript
table now uses three completed USNet retraining seeds and an independent
evaluation of the released OFF-Net checkpoint. Validation selects the
checkpoint; the 2,193 testing images are evaluated only after selection.
All directly compared F/PRE/REC/IoU values use one OFF-Net-style convention:
fixed class argmax, nearest-neighbor restoration to the original `1280 x 720`
label, and one foreground confusion matrix against the unmodified original GT
over the complete test partition. Using the original GT for every method avoids
making label quantization depend on a network's native output stride. A common
101-threshold evaluator on the shared `704 x 1280` input grid additionally
provides MaxF and AP. The authors'
checkpoint cross-check separately records the literal released `test.py` path,
which restores OFF-Net's native 1/4-scale loader target as well as its
prediction; that diagnostic is not mixed into the common local ranking.

## Official source boundary

The baseline networks are imported from authors' repositories rather than
reimplemented here. The ORFD queue supports all four compatible official
graphs (USNet, SNE-RoadSeg, OFF-Net, and RoadFormer); PLARD is intentionally
excluded because its released input path requires a LiDAR-derived ADI that is
not defined by the ORFD release.

| Method | Authors' repository | Pinned commit | Imported graph/recipe |
|---|---|---|---|
| USNet | <https://github.com/morancyc/USNet> | `d761158ad42df7dcb62fa257dd02ce11c85f94a5` | `model/usnet.py`, evidential loss, optimizer and transforms |
| SNE-RoadSeg | <https://github.com/hlwang1124/SNE-RoadSeg> | `5e7900bfd59887634ced687ffe85a73018a38659` | dual-ResNet RoadSeg, SNE, cross-entropy and SGD recipe |
| OFF-Net | <https://github.com/chaytonmin/Off-Road-Freespace-Detection> | `50e63d24836198e8fb5af707e521f414104b4876` | MiT-B2 fusion graph, ORFD loader/SNE, loss and SGD recipe |
| RoadFormer | <https://github.com/LiJiahang617/Road-Former> | `f675a3467cb168ebc727648390c304279bbcb079` | official ORFD config, TwinConvNeXt-B, matching losses and decoder |

`tools/fetch_matched_baseline_sources.sh` clones only these authors'
repositories, checks the remote, refuses a dirty target, and checks out the
exact detached commits. Each result records the remote, commit, hashes of the
imported source files, checkpoint hash, full command, package versions, and GPU.
`docs/ral/table1_matched_baselines/source_provenance.json` records the
corresponding Git blobs and SHA-256 values.

The local code supplies only the data adapter, deterministic seed plumbing,
checkpoint/result format, and common evaluator. It does not replace a baseline
backbone, fusion module, decoder, loss, or optimizer.

Initialization also follows the files selected by the official graphs. USNet
loads torchvision's ImageNet ResNet-18 weights (SHA-256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`),
SNE-RoadSeg loads torchvision's ImageNet ResNet-152 weights (SHA-256
`394f9c45966e3651a89bbb78a48410a6755854ce4a5ab64927cf1c7247f85e58`),
and RoadFormer loads the ConvNeXt-Base ImageNet-21K URL embedded in its official
ORFD config (downloaded-file SHA-256
`262fd0376855955f20f6c036aa882f5cb22b88333b766b0fa20174339c11d70d`).
OFF-Net uses its released Kaiming initialization for any optional local
retraining. The current Table III snapshot instead reports an independently
evaluated authors' released checkpoint; it is not presented as a local
multi-seed retraining result.

PLARD is retained in the matched KITTI table but is not adapted to ORFD. Its
official trainable path consumes LiDAR-derived ADI, whereas the released ORFD
partitions provide registered dense depth and no official PLARD-compatible ADI
construction path. Treating dense depth as ADI would change the method's input
definition and weaken reproducibility, so the ORFD comparison includes only
methods whose official graph has an executable compatible input path.

## Environments and source checkout

Create the two pinned environments from the archived specifications. USNet and
SNE-RoadSeg run in the main environment; OFF-Net and RoadFormer run in the
legacy OpenMMLab-compatible environment required by their released graphs.

```bash
conda env create -f configs/environments/litevilnet_ral.yml
conda env create -f configs/environments/litevilnet_roadformer_ral.yml
bash tools/fetch_matched_baseline_sources.sh
```

Run the remaining commands from the LiteViLNet repository root. The fetch
script verifies the official remotes and pinned commits listed above before a
source tree can be used.

## Continuing independent seeds

The formal queue keeps each method's recipe unchanged. If a run is
interrupted, restart it with the same output directory and `--resume`; the
loader reconstructs completed physical and optimizer-step counters. The
capacity-aware continuation helpers are:

```bash
# Resume OFF-Net seeds 40, 41, and 42
setsid -f bash tools/dispatch_orfd_offnet_resume.sh

# Capacity-aware continuation when GPU1 is shared with SNE/RoadFormer
# (seed 40 remains owned by the independent GPU0 watcher)
GPU=1 SEEDS='41 42' setsid -f bash tools/dispatch_orfd_offnet_capacity_queue.sh

# Priority GPU0 continuation when SNE seed40 is deferred
GPU=0 SEEDS='40' setsid -f bash tools/dispatch_orfd_offnet_resume.sh

# Resume SNE-RoadSeg seeds 40, 41, and 42 (single-owner high-memory queue)
setsid -f bash tools/dispatch_orfd_sne_capacity_queue.sh

# Resume USNet seed 42 (the small AMP job)
setsid -f bash tools/dispatch_orfd_usnet42_resume.sh

# RoadFormer seed 40--42 on GPU 0
setsid -f bash tools/dispatch_orfd_roadformer_after_sne.sh

# Deferred SNE seed 42 after the priority OFF-Net GPU1 lane
setsid -f bash tools/dispatch_orfd_sne42_after_offnet.sh

# Resume SNE seed40 after the priority OFF-Net GPU0 lane
setsid -f bash tools/dispatch_orfd_sne40_after_offnet.sh

```

They claim a seed only when its result is absent and no matching training
process exists. GPU placement is selected by the scripts' `GPU` variable;
the default continuation uses GPU 1 for SNE seeds and GPU 0 for RoadFormer.
The helpers wait for reported memory capacity and do not alter batch size,
epoch count, precision, validation interval, or input resolution. Run the
strict summarizer only after all requested `result.json` files exist. For a
long queue, `setsid -f bash tools/monitor_orfd_partial_summaries.sh` emits
separate `summary_<method>.{json,csv}` snapshots for USNet, SNE-RoadSeg,
OFF-Net, and RoadFormer as soon as each method's three seeds finish; it never
replaces the final all-method `summary.json`.

RoadFormer's formal ORFD command remains FP32, batch 4, 50 epochs, and
`704x1280`. The runner uses activation checkpointing plus PyTorch's
`save_on_cpu` context only to keep saved backward tensors out of the 48-GB
allocator, and enables the official TwinConvNeXt `with_cp` checkpoint switch;
it does not change parameters, optimizer steps, data order, or evaluation.
`result.json` records these memory safeguards and the allocator configuration
for exact reproduction. The default allocator setting is
`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32`.

## Data and geometry preparation

Obtain the official ORFD `Final_Dataset` through the dataset links maintained
in the authors' repository
<https://github.com/chaytonmin/Off-Road-Freespace-Detection#dataset>, then
verify the archive hash below before extraction. The released counts are:

```text
training    8392
validation  1245
testing     2193
```

The complete archive contains registered RGB, dense depth, freespace labels,
and a timestamp-matched calibration file for every sample. ORFD has two
released intrinsic matrices separated by collection time. The cache tool can
record a nearest-timestamp fallback when diagnosing an incomplete extraction,
but every formal command enables `--require-exact-calibration` and therefore
rejects that condition before any cache is generated. Exact/inferred counts
and the maximum timestamp gap are recorded in `normal_cache_metadata.json`; no
calibration is estimated from labels.

The official ZIP used for the formal reproduction has SHA-256
`02359e4b569b12766e317097d84d45d8b9609d8eccea63a9e6c0024e9a2dc92b`;
`unzip -tq` reports no errors. After extraction, RGB, dense depth, GT, and
calibration counts each match 8,392/1,245/2,193, with no RGB--depth stem
differences.

Two normal caches are intentionally separate:

- USNet, SNE-RoadSeg, and RoadFormer use the pinned SNE-RoadSeg implementation,
  whose released path masks points above the camera horizon.
- OFF-Net uses its own pinned `models/sne_model.py`, whose ORFD path does not
  apply that top-crop operation and divides released dense depth by 256.

This prevents two similarly named but numerically different official SNE
implementations from being mixed. USNet's training flip is cached by flipping
depth before SNE, matching its official transform order. RoadFormer normals are
deterministically encoded as the uint16 PNG representation expected by its
official ORFD loader.

```bash
export ORFD_ROOT=/path/to/Final_Dataset
export SNE_SOURCE="$PWD/third_party/matched_baselines/SNE-RoadSeg"
export OFFNET_SOURCE="$PWD/third_party/matched_baselines/OFF-Net"
export SNE_NORMAL_ROOT=runs/revision_1/matched_orfd/normals/sne_roadseg
export OFFNET_NORMAL_ROOT=runs/revision_1/matched_orfd/normals/offnet

CUDA_VISIBLE_DEVICES=0 conda run -n litevilnet_ral \
  env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  python tools/cache_official_orfd_normals.py \
  --data-root "$ORFD_ROOT" --official-source "$SNE_SOURCE" \
  --output-root "$SNE_NORMAL_ROOT" --profile sne_roadseg \
  --include-flipped-training --require-exact-calibration \
  --workers 1 --device cuda:0

CUDA_VISIBLE_DEVICES=1 conda run -n litevilnet_roadformer_ral \
  env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  python tools/cache_official_orfd_normals.py \
  --data-root "$ORFD_ROOT" --official-source "$OFFNET_SOURCE" \
  --output-root "$OFFNET_NORMAL_ROOT" --profile offnet \
  --require-exact-calibration --workers 1 --device cuda:0

export ORFD_ROADFORMER_ROOT=runs/revision_1/matched_orfd/roadformer_orfd
python tools/prepare_matched_orfd_roadformer.py \
  --data-root "$ORFD_ROOT" --normal-root "$SNE_NORMAL_ROOT" \
  --output-root "$ORFD_ROADFORMER_ROOT" --workers 8
```

The formal caches were generated with the CUDA commands above and then rerun
without `--force`, which validated the shape and dtype of every stored array
and recorded all 20,222/11,830 entries as reused. A CPU-only reconstruction can
instead use `--device cpu` with multiple workers; the pinned computation is the
same, while small backend-dependent floating-point differences are possible.

The prepared RoadFormer tree uses symlinks for released RGB/labels and does not
duplicate the dataset. Generated normal caches and prepared inputs are excluded
from the submission archive and are reproducible with the commands above.

## Training protocol

The formal comparison uses seeds `40`, `41`, and `42` for each compatible
official graph. Every run uses the same released partitions, `704 x 1280`
network inputs, validation 101-threshold MaxF for checkpoint selection, and
the same final test evaluator. RoadFormer's logits are taken before its
generic metadata-based restoration so its threshold sweep uses the same input
grid and its discrete argmax mask follows the same nearest-neighbor original-
GT restoration as the other rows.
Method-specific optimization follows each pinned official recipe.

The completed USNet three-seed summary (percentages, held-out test; mean and
sample SD over seeds 40, 41, and 42) is:

| Seed | F-score | AP | PRE | REC | IoU |
|---:|---:|---:|---:|---:|---:|
| 40 | 95.6188 | 97.1748 | 95.3155 | 95.9241 | 91.6054 |
| 41 | 95.5689 | 97.9032 | 94.4440 | 96.7208 | 91.5137 |
| 42 | 96.6522 | 98.3595 | 96.8555 | 96.4498 | 93.5213 |
| Mean $\pm$ sample SD | $95.9466\pm0.6116$ | $97.8125\pm0.5975$ | $95.5383\pm1.2211$ | $96.3649\pm0.4051$ | $92.2135\pm1.1335$ |

USNet's completed three-seed result is used in the manuscript ORFD table.
SNE-RoadSeg/OFF-Net/RoadFormer local ORFD retraining remains incomplete and is
not inferred from partial runs. The OFF-Net row remains explicitly identified
as one local evaluation of the authors' released checkpoint.

| Method | Epochs | Physical batch | Grad. accum. | Effective batch | Precision | Checkpoint selection |
|---|---:|---:|---:|---:|---|---|
| USNet | 30 | 2 | 1 | 2 | AMP | validation 101-threshold MaxF |
| SNE-RoadSeg | 30 | 2 | 1 | 2 | AMP | validation 101-threshold MaxF |
| OFF-Net | 30 | 2 | 4 | 8 | FP32 | validation 101-threshold MaxF |
| RoadFormer | 50 | 4 | 1 | 4 | FP32 | validation 101-threshold MaxF |

OFF-Net's released command uses global batch 8 over four GPUs, hence each
DataParallel replica sees batch 2. On one GPU we retain that physical batch and
accumulate four consecutive gradients before each update, reproducing effective
batch 8 without changing batch-normalization exposure. RoadFormer uses the
50-epoch schedule and batch 4 in its official ORFD configuration. USNet and
SNE-RoadSeg use the 30-epoch ORFD budget while retaining their official model,
loss, initialization, optimizer, schedule, and applicable augmentations.
The 8,392-image training partition is exactly divisible by both physical batch
sizes used here (2 and 4), so `drop_last=True` does not omit any training image.
Python, NumPy, PyTorch/CUDA, sampler, and worker RNGs are derived from the
recorded run seed before model construction and data loading.
The CPU unit test
`test_offnet_four_microbatches_match_one_global_mean_loss_update` checks that
the accumulated mean-loss gradient and SGD update match a direct batch-8
update.

After the caches and RoadFormer tree are ready, run the capacity-aware queue
below. It leaves unrelated compute processes untouched, avoids duplicate seed
claims, and starts independent SNE seed-41 work on GPU0 once the short USNet
seed-42 job releases that slot:

```bash
export USNET_SOURCE="$PWD/third_party/matched_baselines/USNet"
export ROADFORMER_SOURCE="$PWD/third_party/matched_baselines/Road-Former"
export ORFD_SNE_NORMAL_ROOT="$SNE_NORMAL_ROOT"
export ORFD_OFFNET_NORMAL_ROOT="$OFFNET_NORMAL_ROOT"
export ORFD_ROADFORMER_ROOT
export OUTPUT_ROOT=runs/revision_1/matched_orfd/formal
export GPU0=0 GPU1=1
bash tools/run_matched_orfd_baselines.sh

# Optional long-running follow-up dispatcher (two RTX 4090 D GPUs).
bash tools/dispatch_orfd_followup.sh
```

The queue writes one complete directory per method and seed, then validates all
source commits, counts, input sizes, test sample counts, and parameter counts
before producing:

```text
docs/ral/orfd_matched_baselines/results/
  summary.json
  summary.csv
  seeds/<method>_orfd_seed<seed>.json
```

Large checkpoints remain in `runs/` for local audit. Only anonymous result
copies and reconstruction code enter the submission archive.

For an independent evaluator check, the authors' README links checkpoint
`best_net_RoadSeg.pth` (Google Drive file
`1X53H8QFuiVv1OMTkyCbPZ3Hu0e_oO7sp`, SHA-256
`4d82a9ba1c5411a8159e1a262f1741a6a163c8bd9751d52a90fb10deb237650d`).
After the OFF-Net normal cache is ready, evaluate it with
`tools/evaluate_official_offnet_orfd.py`. The output reports both the common
original-GT metric and the literal authors' 1/4-target round-trip metric. This
cross-check is kept separate from the three independently initialized local
retraining rows. With the exact-calibration cache, strict loading of the
released checkpoint gives 92.80 F, 94.53 PRE, 91.13 REC, 86.57 IoU, 92.92
MaxF, and 97.58 AP under the common original-GT evaluator. The literal
released target round trip gives 92.85 F, 94.52 PRE, 91.24 REC, and 86.66 IoU.
The complete counts and unrounded values are stored in
`runs/revision_1/matched_orfd/official_offnet_checkpoint_test_exact.json` and
an anonymized copy is included in the complete reproduction archive.

## KITTI OFF-Net row

OFF-Net is also retrained on the same matched KITTI protocol used by Table I:
the fixed 231/58 category-stratified split, `384 x 1248`, 150 epochs, per-GPU
batch 2, seeds 40/41/42, and the common 101-threshold perspective-view MaxF
evaluator. KITTI normals are separately generated by OFF-Net's SNE:

```bash
conda run -n litevilnet_roadformer_ral env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  python tools/cache_official_sne_normals.py \
  --data-root "$MATCHED_ROOT" --official-source "$OFFNET_SOURCE" \
  --output-root "$KITTI_OFFNET_NORMAL_ROOT" --profile offnet --workers 4

bash tools/run_matched_kitti_offnet.sh
```

`tools/run_matched_kitti_fps.sh` then measures OFF-Net with the same RTX 4090 D,
batch-1, `384 x 1248`, PyTorch FP32, resident-input, model-only CUDA-event
protocol used for every Table-I row.
