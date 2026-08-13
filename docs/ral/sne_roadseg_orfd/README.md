# Anonymous ORFD SNE-RoadSeg Reproduction Supplement

This supplement documents the completed SNE-RoadSeg comparison on the official
ORFD held-out test partition. It contains the official-source provenance,
portable training and evaluation instructions, seed-level result records, and
the aggregation output used for the manuscript revision. The archive excludes
the ORFD data, generated normal cache, model checkpoints, third-party source
trees, version-control metadata, local absolute paths, and author identity.

## Evidence scope

SNE-RoadSeg is evaluated from the authors' official repository at the pinned
commit in `source_provenance.json`. The local adapter does not reimplement its
network, loss, optimizer, or normal estimation. It only supplies the released
ORFD split adapter, deterministic seed plumbing, checkpoint selection, and the
common held-out evaluator.

The formal protocol is unchanged across seeds 40, 41, and 42:

- official ORFD train/validation/test partitions (8,392/1,245/2,193 pairs);
- `704 x 1280` network input and original `1280 x 720` ground-truth evaluation;
- 30 epochs, physical batch size 2, AMP, and validation 101-threshold MaxF
  checkpoint selection;
- fixed argmax/0.5 predictions resized with nearest-neighbor interpolation,
  followed by one confusion matrix over all held-out test pixels;
- threshold-swept AP and MaxF are retained as complementary ranking metrics.

The three seed records are stored under
`results/seeds_sne_roadseg/`; `results/summary_sne_roadseg.{json,csv}` contains
the mean and sample standard deviation. `RESULTS.md` is a compact, portable
human-readable summary generated from the same strict aggregate. Paths in
these archived copies use
portable placeholders and do not alter metrics or hashes.

## Environment and data

From the LiteViLNet repository root:

```bash
conda env create -f configs/environments/litevilnet_ral.yml
conda activate litevilnet_ral
```

Obtain ORFD through the dataset link maintained by the official OFF-Net
repository and set portable roots. Generate the SNE normal cache using the
official SNE-RoadSeg implementation; the formal cache requires exact released
calibration matches:

```bash
export ORFD_ROOT=/path/to/Final_Dataset
export SNE_SOURCE=/path/to/SNE-RoadSeg
export SNE_NORMAL_ROOT=runs/revision_1/matched_orfd/local_exact_normals/sne_roadseg

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n litevilnet_ral \
  env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  python tools/cache_official_orfd_normals.py \
  --data-root "$ORFD_ROOT" --official-source "$SNE_SOURCE" \
  --output-root "$SNE_NORMAL_ROOT" --profile sne_roadseg \
  --include-flipped-training --require-exact-calibration \
  --workers 1 --device cuda:0
```

The cache generator records the official source hash, calibration-match count,
depth scale, and task count. It can also run on CPU with the same numerical
recipe when GPU memory is unavailable.

## Training and evaluation

Run one seed at a time or schedule the three seeds on separate GPUs. The
following command is the exact formal command with portable paths:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n litevilnet_ral \
  env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  python tools/train_matched_orfd_baseline.py \
  --baseline sne_roadseg --official-source "$SNE_SOURCE" \
  --data-root "$ORFD_ROOT" --normal-root "$SNE_NORMAL_ROOT" \
  --output-dir runs/revision_1/matched_orfd/formal/sne_roadseg_seed40 \
  --seed 40 --epochs 30 --batch-size 2 --num-workers 8 \
  --height 704 --width 1280 --val-every 1 --amp --device cuda
```

Replace `40` in the output directory and `--seed` with `41` and `42`. Do not
change the batch size, epoch budget, input size, AMP setting, or evaluator when
reproducing the reported rows. After all three `result.json` files exist:

```bash
python tools/summarize_matched_orfd_baselines.py \
  --input-root runs/revision_1/matched_orfd/formal \
  --expected-seeds 40,41,42 --methods sne_roadseg \
  --output-json docs/ral/orfd_matched_baselines/results/summary_sne_roadseg.json \
  --output-csv docs/ral/orfd_matched_baselines/results/summary_sne_roadseg.csv \
  --seed-output-dir docs/ral/orfd_matched_baselines/results/seeds_sne_roadseg
```

The summarizer verifies split counts, input grids, official commit and source
hashes, parameter count, checkpoint hashes, calibration metadata, and complete
test-pixel coverage before writing the aggregate.

Generate the human-readable result sheet used for author verification:

```bash
python tools/write_sne_roadseg_report.py \
  --summary docs/ral/orfd_matched_baselines/results/summary_sne_roadseg.json \
  --output docs/ral/sne_roadseg_orfd/RESULTS.md
```

## Anonymous packaging

Use `tools/package_ral_sne_roadseg.sh` from the repository root. It rebuilds
sanitized temporary copies, scans paths and contents for identity leakage,
normalizes archive ownership and timestamps, and writes a SHA-256 sidecar. Set
`LITEVILNET_DOUBLE_BLIND_TOKENS` to any private identity strings that should be
rejected by the scan. The original result files are never rewritten.
