#!/usr/bin/env python
"""Benchmark a TensorRT engine through trtexec and record RA-L metrics."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from litevilnet.utils.common import append_csv, run_command, system_metadata, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TensorRT engine with trtexec")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", default="VLLiNet")
    parser.add_argument("--preset", default="")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--output", default="runs/tensorrt/tensorrt_benchmark.json")
    parser.add_argument("--csv", default="runs/benchmark/benchmark_summary.csv")
    parser.add_argument("--extra", nargs="*", default=[])
    return parser.parse_args()


def parse_trtexec_output(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    patterns = {
        "mean_ms": r"Latency:.*?mean = ([0-9.]+) ms",
        "p50_ms": r"Latency:.*?median = ([0-9.]+) ms",
        "p95_ms": r"Latency:.*?percentile\(95%\) = ([0-9.]+) ms",
        "throughput_qps": r"Throughput: ([0-9.]+) qps",
        "gpu_compute_mean_ms": r"GPU Compute Time:.*?mean = ([0-9.]+) ms",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.S)
        if match:
            metrics[key] = float(match.group(1))
    if "mean_ms" in metrics:
        metrics["fps"] = 1000.0 / metrics["mean_ms"]
    elif "throughput_qps" in metrics:
        metrics["fps"] = metrics["throughput_qps"]
    return metrics


def main() -> None:
    args = parse_args()
    trtexec = shutil.which(args.trtexec)
    if trtexec is None:
        raise SystemExit("trtexec not found. Install TensorRT or pass --trtexec /path/to/trtexec")

    engine = Path(args.engine)
    command = [
        trtexec,
        f"--loadEngine={engine}",
        f"--warmUp={args.warmup}",
        f"--duration={args.duration}",
        "--useCudaGraph",
        "--separateProfileRun",
    ]
    command.extend(args.extra)
    return_code, output = run_command(command)
    parsed = parse_trtexec_output(output)
    result = {
        "backend": "tensorrt",
        "model": args.model,
        "preset": args.preset,
        "precision": args.precision,
        "engine": str(engine),
        "engine_size_mb": engine.stat().st_size / (1024 * 1024) if engine.exists() else None,
        "return_code": return_code,
        "metrics": parsed,
        "command": command,
        "trtexec_output": output,
        "system": system_metadata(),
    }
    write_json(args.output, result)
    append_csv(
        args.csv,
        {
            "backend": "tensorrt",
            "model": args.model,
            "preset": args.preset,
            "precision": args.precision,
            "batch_size": 1,
            "img_h": "",
            "img_w": "",
            "mean_ms": parsed.get("mean_ms", ""),
            "p95_ms": parsed.get("p95_ms", ""),
            "fps": parsed.get("fps", ""),
            "parameters_M": "",
            "memory_mb": "",
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
