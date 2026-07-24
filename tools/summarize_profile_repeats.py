#!/usr/bin/env python
"""Aggregate repeated ``profile_ablation`` JSON files for paper tables."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from litevilnet.utils.common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize repeated model-only profiles")
    parser.add_argument("profiles", nargs="+", help="JSON outputs from tools.profile_ablation")
    parser.add_argument("--cost-json", default="", help="Optional profile JSON containing fvcore cost")
    parser.add_argument("--output", default="runs/revision_1/ablation_profile_summary.json")
    return parser.parse_args()


def sample_stats(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.profiles]
    if not payloads:
        raise SystemExit("No profile files supplied")

    signature_keys = ("input_shape_per_modality", "precision", "warmup", "iterations")
    reference_protocol = payloads[0]["protocol"]
    for path, payload in zip(args.profiles[1:], payloads[1:]):
        mismatched = [
            key for key in signature_keys
            if payload["protocol"].get(key) != reference_protocol.get(key)
        ]
        if mismatched:
            raise SystemExit(f"Protocol mismatch in {path}: {mismatched}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for payload in payloads:
        for result in payload["results"]:
            if "cuda" not in result:
                raise SystemExit(f"Missing CUDA timing for {result['config']}")
            grouped[result["config"]].append(result)

    costs = {}
    if args.cost_json:
        cost_payload = json.loads(Path(args.cost_json).read_text(encoding="utf-8"))
        costs = {result["config"]: result.get("cost") for result in cost_payload["results"]}

    summaries = []
    for config, records in sorted(grouped.items()):
        parameter_values = {int(record["parameters"]) for record in records}
        if len(parameter_values) != 1:
            raise SystemExit(f"Parameter mismatch across {config} repeats: {parameter_values}")
        mean_latency = [float(record["cuda"]["latency_ms"]["mean_ms"]) for record in records]
        p95_latency = [float(record["cuda"]["latency_ms"]["p95_ms"]) for record in records]
        peak_memory = [float(record["cuda"]["cuda_peak_allocated_mb"]) for record in records]
        incremental_memory = [float(record["cuda"]["cuda_incremental_peak_mb"]) for record in records]
        summaries.append(
            {
                "config": config,
                "parameters": parameter_values.pop(),
                "repeat_mean_latency_ms": sample_stats(mean_latency),
                "repeat_p95_latency_ms": sample_stats(p95_latency),
                "cuda_peak_allocated_mb": sample_stats(peak_memory),
                "cuda_incremental_peak_mb": sample_stats(incremental_memory),
                "cost": costs.get(config),
            }
        )

    output = {
        "protocol": {
            **{key: reference_protocol.get(key) for key in signature_keys},
            "aggregation": "mean and sample SD across independent profiler invocations",
            "latency_scope": payloads[0]["results"][0]["cuda"].get("scope"),
        },
        "source_files": args.profiles,
        "cost_source": args.cost_json or None,
        "systems": [payload.get("system") for payload in payloads],
        "nvidia_compute_processes_before": [
            payload.get("nvidia_compute_processes_before") for payload in payloads
        ],
        "nvidia_compute_processes_after": [
            payload.get("nvidia_compute_processes_after") for payload in payloads
        ],
        "results": summaries,
    }
    write_json(args.output, output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
