#!/usr/bin/env python3
"""Write a portable, anonymous SNE-RoadSeg ORFD result summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(value: float) -> str:
    return f"{100.0 * value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    method = summary["methods"]["sne_roadseg"]
    if method["seeds"] != [40, 41, 42]:
        raise RuntimeError("Expected formal SNE-RoadSeg seeds 40, 41, 42")

    records = []
    root = args.summary.parents[4] / "runs/revision_1/matched_orfd/formal"
    for seed in method["seeds"]:
        path = root / f"sne_roadseg_seed{seed}" / "result.json"
        item = json.loads(path.read_text(encoding="utf-8"))["testing"]
        fixed = item["official_fixed_argmax"]
        swept = item["threshold_swept"]
        records.append(
            f"| {seed} | {pct(fixed['F_score'])} | {pct(swept['AP'])} | "
            f"{pct(fixed['PRE'])} | {pct(fixed['REC'])} | {pct(fixed['IoU'])} |"
        )

    fixed = method["testing_fixed_argmax"]
    swept = method["testing_threshold_swept"]
    report = f"""# SNE-RoadSeg ORFD Formal Results

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
{chr(10).join(records)}
| Mean +/- sample SD | {pct(fixed['F_score']['mean'])}+/-{pct(fixed['F_score']['sample_sd'])} | {pct(swept['AP']['mean'])}+/-{pct(swept['AP']['sample_sd'])} | {pct(fixed['PRE']['mean'])}+/-{pct(fixed['PRE']['sample_sd'])} | {pct(fixed['REC']['mean'])}+/-{pct(fixed['REC']['sample_sd'])} | {pct(fixed['IoU']['mean'])}+/-{pct(fixed['IoU']['sample_sd'])} |

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
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
