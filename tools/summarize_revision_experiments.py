#!/usr/bin/env python
"""Aggregate seed-level revision results without hand-copying paper numbers."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from litevilnet.utils.common import ensure_parent, write_json


METRICS = ("MaxF", "AP", "PRE", "REC", "FPR", "FNR", "IoU", "BestThreshold")
PROTOCOL_FIELDS = (
    "data_root",
    "train_split_file",
    "val_split_file",
    "train_samples",
    "val_samples",
    "image_height",
    "image_width",
    "batch_size",
    "accumulate_grad_batches",
    "learning_rate",
    "weight_decay",
    "drop_last",
    "deterministic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize deterministic multi-seed experiments")
    parser.add_argument("roots", nargs="+", help="Experiment roots containing CONFIG/seed_N/result.json")
    parser.add_argument("--output", default="runs/revision_1/experiment_summary.json")
    parser.add_argument("--csv", default="runs/revision_1/experiment_summary.csv")
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="LEFT:RIGHT",
        help=(
            "Also summarize seed-matched metric differences LEFT minus RIGHT. "
            "May be specified more than once, for example --pair full:optimal."
        ),
    )
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    groups = defaultdict(list)
    for root_string in args.roots:
        root = Path(root_string)
        for result_path in sorted(root.glob("*/seed_*/result.json")):
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            dataset = payload.get("dataset") or ("orfd" if "orfd" in str(root).lower() else "kitti")
            groups[(dataset, payload["config"])].append((payload, result_path))
    if not groups:
        raise SystemExit("No CONFIG/seed_N/result.json files found")

    summaries = []
    rows = []
    for (dataset, config), records in sorted(groups.items()):
        records.sort(key=lambda item: int(item[0]["seed"]))
        seeds = [int(payload["seed"]) for payload, _ in records]
        if len(seeds) != len(set(seeds)):
            raise SystemExit(f"Duplicate seeds for {dataset}/{config}: {seeds}")
        parameter_values = {int(payload["params"]) for payload, _ in records}
        if len(parameter_values) != 1:
            raise SystemExit(
                f"Parameter mismatch for {dataset}/{config}: {sorted(parameter_values)}"
            )
        protocol = {}
        for field in PROTOCOL_FIELDS:
            values = {json.dumps(payload.get(field), sort_keys=True) for payload, _ in records}
            if len(values) != 1:
                raise SystemExit(
                    f"Protocol mismatch for {dataset}/{config}, field {field}: {sorted(values)}"
                )
            protocol[field] = records[0][0].get(field)
        summary = {
            "dataset": dataset,
            "config": config,
            "seeds": seeds,
            "parameters": parameter_values.pop(),
            "protocol": protocol,
            "metrics": {},
            "source_files": [str(path) for _, path in records],
        }
        row = {
            "dataset": dataset,
            "config": config,
            "seeds": ";".join(str(seed) for seed in summary["seeds"]),
            "parameters_M": summary["parameters"] / 1e6,
        }
        for metric in METRICS:
            values = [float(payload["best_val_metrics"][metric]) for payload, _ in records]
            metric_stats = stats(values)
            summary["metrics"][metric] = metric_stats
            row[f"{metric}_mean"] = metric_stats["mean"]
            row[f"{metric}_sample_std"] = metric_stats["sample_std"]
        summaries.append(summary)
        rows.append(row)

    requested_pairs = []
    for pair in args.pair:
        if pair.count(":") != 1:
            raise SystemExit(f"Invalid --pair {pair!r}; expected LEFT:RIGHT")
        left, right = pair.split(":", 1)
        if not left or not right:
            raise SystemExit(f"Invalid --pair {pair!r}; expected LEFT:RIGHT")
        requested_pairs.append((left, right))

    paired_comparisons = []
    datasets = sorted({dataset for dataset, _ in groups})
    for dataset in datasets:
        for left, right in requested_pairs:
            left_records = {
                int(payload["seed"]): payload
                for payload, _ in groups.get((dataset, left), [])
            }
            right_records = {
                int(payload["seed"]): payload
                for payload, _ in groups.get((dataset, right), [])
            }
            common_seeds = sorted(set(left_records) & set(right_records))
            if not common_seeds:
                continue
            comparison = {
                "dataset": dataset,
                "left": left,
                "right": right,
                "definition": f"{left} minus {right}",
                "seeds": common_seeds,
                "metrics": {},
            }
            for metric in METRICS:
                differences = [
                    float(left_records[seed]["best_val_metrics"][metric])
                    - float(right_records[seed]["best_val_metrics"][metric])
                    for seed in common_seeds
                ]
                comparison["metrics"][metric] = {
                    **stats(differences),
                    "per_seed": dict(zip(map(str, common_seeds), differences)),
                }
            paired_comparisons.append(comparison)

    output_payload = {
        "groups": summaries,
        "paired_comparisons": paired_comparisons,
    }
    write_json(args.output, output_payload)
    csv_path = ensure_parent(args.csv)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
