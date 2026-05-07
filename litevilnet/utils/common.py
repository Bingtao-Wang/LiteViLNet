"""Shared helpers for VLLiNet deployment scripts."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = ensure_parent(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((q / 100.0) * (len(values) - 1)))))
    return values[idx]


def summarize_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "fps": 0.0}
    mean_ms = sum(values) / len(values)
    return {
        "mean_ms": mean_ms,
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "min_ms": min(values),
        "max_ms": max(values),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }


def system_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if torch.cuda.is_available():
        metadata.update(
            {
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    for path in ("/etc/nv_tegra_release", "/proc/device-tree/model"):
        if os.path.exists(path):
            try:
                metadata[Path(path).name.replace("-", "_")] = Path(path).read_text(errors="ignore").strip("\x00\n ")
            except OSError:
                pass
    return metadata


def run_command(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return result.returncode, result.stdout
    except FileNotFoundError as exc:
        return 127, str(exc)

