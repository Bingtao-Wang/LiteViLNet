#!/usr/bin/env python
"""Summarize multi-seed held-out ORFD testing evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

from litevilnet.utils.common import ensure_parent, write_json


METRICS = ("MaxF", "AP", "PRE", "REC", "FPR", "FNR", "IoU", "BestThreshold")
OFFICIAL_METRICS = ("F_score", "PRE", "REC", "IoU", "GlobalAcc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="runs/revision_1/orfd_test")
    parser.add_argument("--output-json", default="runs/revision_1/orfd_test_summary.json")
    parser.add_argument("--output-csv", default="runs/revision_1/orfd_test_summary.csv")
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def seed_from_checkpoint(checkpoint: str) -> int:
    match = re.search(r"seed_(\d+)", checkpoint)
    if not match:
        raise ValueError(f"Cannot recover seed from checkpoint path: {checkpoint}")
    return int(match.group(1))


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for path in sorted(Path(args.input_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("partition") != "test":
            continue
        payload["seed"] = seed_from_checkpoint(payload["checkpoint"])
        payload["source_file"] = str(path.resolve())
        records.append(payload)
    if not records:
        raise FileNotFoundError(f"No ORFD testing result JSONs found in {args.input_dir}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["config"], []).append(record)

    groups = []
    for config, items in sorted(grouped.items()):
        items.sort(key=lambda item: item["seed"])
        groups.append(
            {
                "dataset": "ORFD",
                "partition": "test",
                "config": config,
                "seeds": [item["seed"] for item in items],
                "samples": items[0]["samples"],
                "parameters": items[0]["parameters"],
                "metrics": {
                    metric: summarize([float(item["metrics"][metric]) for item in items])
                    for metric in METRICS
                },
                "official_metrics": {
                    metric: summarize([float(item["official_metrics"][metric]) for item in items])
                    for metric in OFFICIAL_METRICS
                },
                "checkpoint_sha256": {
                    str(item["seed"]): item["checkpoint_sha256"] for item in items
                },
                "source_files": [item["source_file"] for item in items],
            }
        )

    paired = []
    if "full" in grouped and "optimal" in grouped:
        full = {item["seed"]: item for item in grouped["full"]}
        optimal = {item["seed"]: item for item in grouped["optimal"]}
        seeds = sorted(set(full) & set(optimal))
        paired_metrics = {}
        for metric in METRICS:
            per_seed = {
                str(seed): float(full[seed]["metrics"][metric])
                - float(optimal[seed]["metrics"][metric])
                for seed in seeds
            }
            paired_metrics[metric] = {
                **summarize(list(per_seed.values())),
                "per_seed": per_seed,
            }
        official_paired_metrics = {}
        for metric in OFFICIAL_METRICS:
            per_seed = {
                str(seed): float(full[seed]["official_metrics"][metric])
                - float(optimal[seed]["official_metrics"][metric])
                for seed in seeds
            }
            official_paired_metrics[metric] = {
                **summarize(list(per_seed.values())),
                "per_seed": per_seed,
            }
        paired.append(
            {
                "left": "full",
                "right": "optimal",
                "definition": "left minus right on the same seed",
                "seeds": seeds,
                "metrics": paired_metrics,
                "official_metrics": official_paired_metrics,
            }
        )

    output = {
        "protocol": {
            "dataset": "ORFD released archive",
            "partition": "testing",
            "selection": "checkpoints selected only on the released validation partition",
            "samples": records[0]["samples"],
            "input_resolution": [704, 1280],
            "precision": records[0]["precision"],
            "thresholds": records[0]["thresholds"],
            "official_metric": "fixed argmax/0.5 prediction resized with nearest-neighbor to original 1280x720 GT, following OFF-Net commit 50e63d2",
            "split_overlap": {"train_test_filenames": 0, "validation_test_filenames": 0},
        },
        "groups": groups,
        "paired_comparisons": paired,
    }
    write_json(args.output_json, output)

    csv_path = ensure_parent(args.output_csv)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["config", "seeds", "samples", "parameters"]
        for metric in METRICS:
            fieldnames.extend((f"{metric}_mean", f"{metric}_sample_std"))
        for metric in OFFICIAL_METRICS:
            fieldnames.extend((f"official_{metric}_mean", f"official_{metric}_sample_std"))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            row: dict[str, Any] = {
                "config": group["config"],
                "seeds": ",".join(str(seed) for seed in group["seeds"]),
                "samples": group["samples"],
                "parameters": group["parameters"],
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = group["metrics"][metric]["mean"]
                row[f"{metric}_sample_std"] = group["metrics"][metric]["sample_std"]
            for metric in OFFICIAL_METRICS:
                row[f"official_{metric}_mean"] = group["official_metrics"][metric]["mean"]
                row[f"official_{metric}_sample_std"] = group["official_metrics"][metric]["sample_std"]
            writer.writerow(row)

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
