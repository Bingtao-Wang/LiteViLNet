#!/usr/bin/env python
"""Summarize multi-seed KD students against seed-matched non-KD students."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from litevilnet.utils.common import write_json


METRICS = ("MaxF", "AP", "PRE", "REC", "FPR", "FNR", "IoU", "BestThreshold")
SEED_DIRECTORY = re.compile(r"seed_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distill-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-config", default="add_lidar")
    parser.add_argument("--output", default="runs/revision_1/kitti_distillation_summary.json")
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def load_distillation_results(root: Path) -> dict[int, tuple[dict, Path]]:
    records = {}
    for directory in sorted(root.iterdir()):
        match = SEED_DIRECTORY.fullmatch(directory.name)
        if match is None:
            continue
        result_path = directory / "result.json"
        if not result_path.is_file():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        seed = int(match.group(1))
        if int(payload["seed"]) != seed:
            raise ValueError(f"Seed mismatch in {result_path}")
        records[seed] = (payload, result_path)
    return records


def load_baselines(root: Path, config: str) -> dict[int, tuple[dict, Path]]:
    records = {}
    for result_path in sorted((root / config).glob("seed_*/result.json")):
        match = SEED_DIRECTORY.fullmatch(result_path.parent.name)
        if match is None:
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        seed = int(match.group(1))
        if payload["config"] != config or int(payload["seed"]) != seed:
            raise ValueError(f"Config/seed mismatch in {result_path}")
        records[seed] = (payload, result_path)
    return records


def main() -> None:
    args = parse_args()
    distilled = load_distillation_results(Path(args.distill_root))
    baselines = load_baselines(Path(args.baseline_root), args.baseline_config)
    common_seeds = sorted(set(distilled) & set(baselines))
    if not common_seeds:
        raise SystemExit("No complete seed-matched KD/baseline results")

    output = {
        "definition": "KD student minus same-architecture non-KD student",
        "baseline_config": args.baseline_config,
        "seeds": common_seeds,
        "kd_source_files": [str(distilled[seed][1]) for seed in common_seeds],
        "baseline_source_files": [str(baselines[seed][1]) for seed in common_seeds],
        "kd_metrics": {},
        "baseline_metrics": {},
        "paired_differences": {},
        "teacher": distilled[common_seeds[0]][0]["teacher"],
        "student": distilled[common_seeds[0]][0]["student"],
    }
    for metric in METRICS:
        kd_values = [
            float(distilled[seed][0]["best_val_metrics"][metric])
            for seed in common_seeds
        ]
        baseline_values = [
            float(baselines[seed][0]["best_val_metrics"][metric])
            for seed in common_seeds
        ]
        differences = [
            kd_value - baseline_value
            for kd_value, baseline_value in zip(kd_values, baseline_values)
        ]
        output["kd_metrics"][metric] = stats(kd_values)
        output["baseline_metrics"][metric] = stats(baseline_values)
        output["paired_differences"][metric] = {
            **stats(differences),
            "per_seed": dict(zip(map(str, common_seeds), differences)),
        }

    write_json(args.output, output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
