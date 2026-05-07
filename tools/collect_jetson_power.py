#!/usr/bin/env python
"""Collect Jetson power samples with tegrastats."""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

from litevilnet.utils.common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Jetson tegrastats power samples")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--interval_ms", type=int, default=200)
    parser.add_argument("--output", default="deployment/results/jetson_power.json")
    return parser.parse_args()


def parse_power_mw(line: str) -> float | None:
    values = [float(v) for v in re.findall(r"(?:POM_5V_GPU|POM_5V_CPU|VDD_GPU_SOC|VDD_CPU_CV|VDD_IN)\s+([0-9.]+)mW", line)]
    if values:
        return sum(values)
    match = re.search(r"([0-9.]+)mW", line)
    return float(match.group(1)) if match else None


def main() -> None:
    args = parse_args()
    command = ["tegrastats", "--interval", str(args.interval_ms)]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    samples = []
    started = time.time()
    try:
        while time.time() - started < args.duration:
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                continue
            power_mw = parse_power_mw(line)
            samples.append({"t_s": time.time() - started, "power_mw": power_mw, "raw": line.strip()})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    valid = [s["power_mw"] for s in samples if s["power_mw"] is not None]
    summary = {
        "duration_s": args.duration,
        "interval_ms": args.interval_ms,
        "samples": samples,
        "mean_power_w": (sum(valid) / len(valid) / 1000.0) if valid else None,
        "max_power_w": (max(valid) / 1000.0) if valid else None,
    }
    write_json(args.output, summary)
    print(f"Saved power samples: {Path(args.output)}")


if __name__ == "__main__":
    main()
