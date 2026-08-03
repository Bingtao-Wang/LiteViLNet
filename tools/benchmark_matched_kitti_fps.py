#!/usr/bin/env python3
"""Benchmark a matched Table-I model with one auditable CUDA timing protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXPECTED_COMMITS = {
    "usnet": "d761158ad42df7dcb62fa257dd02ce11c85f94a5",
    "sne_roadseg": "5e7900bfd59887634ced687ffe85a73018a38659",
    "plard": "44485803092e729661c696ab6c03f6f2fabc8701",
    "roadformer": "f675a3467cb168ebc727648390c304279bbcb079",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=("litevilnet", "usnet", "sne_roadseg", "plard", "roadformer"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-source", type=Path)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 1.13 used by the pinned RoadFormer stack.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload)!r}")
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary")
    return {"payload": payload, "state": state}


def verify_source(method: str, source: Path | None) -> dict[str, str] | None:
    if method == "litevilnet":
        return None
    if source is None:
        raise ValueError(f"--official-source is required for {method}")
    source = source.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != EXPECTED_COMMITS[method]:
        raise RuntimeError(f"Unexpected {method} commit: {commit}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip():
        raise RuntimeError(f"Official source is not clean: {source}")
    return {
        "remote": subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=source, text=True
        ).strip(),
        "commit": commit,
    }


def build_standard_model(
    method: str, source: Path | None, checkpoint: Path, device: torch.device
) -> tuple[torch.nn.Module, Callable[[torch.Tensor, torch.Tensor], Any]]:
    if method == "litevilnet":
        from litevilnet.models.vllinet_ablation import get_ablation_model

        model = get_ablation_model("full", pretrained=False)
    else:
        from tools.train_matched_kitti_baseline import load_plard, load_sne_roadseg, load_usnet

        loader = {
            "usnet": load_usnet,
            "sne_roadseg": load_sne_roadseg,
            "plard": load_plard,
        }[method]
        load_args = SimpleNamespace(
            official_source=source,
            learning_rate=None,
            epochs=150,
        )
        model = loader(load_args, device).model

    checkpoint_payload = load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint_payload["state"], strict=True)
    model.to(device).eval()

    def forward(rgb: torch.Tensor, geometry: torch.Tensor) -> Any:
        return model([rgb, geometry]) if method == "plard" else model(rgb, geometry)

    return model, forward


def build_roadformer(
    source: Path, checkpoint: Path, device: torch.device, height: int, width: int
) -> tuple[torch.nn.Module, Callable[[torch.Tensor, torch.Tensor], Any], dict[str, Any]]:
    from tools.train_matched_kitti_roadformer import bootstrap_official

    bridge = bootstrap_official(source)
    from mmengine_custom.config import Config
    from mmseg_custom.registry import MODELS

    config_path = source / "configs" / "roadformer_kitti" / "roadformer_convnext-b_kitti-384x1280.py"
    config = Config.fromfile(config_path)
    config.model.data_preprocessor.size = (height, width)
    config.model.decode_head.pixel_decoder.img_scale = (height, width)
    model = MODELS.build(config.model)
    checkpoint_payload = load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint_payload["state"], strict=True)
    model.to(device).eval()
    metadata = [
        {
            "ori_shape": (height, width),
            "img_shape": (height, width),
            "pad_shape": (height, width),
            "padding_size": [0, 0, 0, 0],
        }
    ]

    def forward(rgb: torch.Tensor, geometry: torch.Tensor) -> Any:
        return model.encode_decode(torch.cat((rgb, geometry), dim=1), metadata)

    return model, forward, bridge


def sample_stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean_ms": float(array.mean()),
        "sample_std_ms": float(array.std(ddof=1)) if len(values) > 1 else 0.0,
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
        "fps_from_mean": float(1000.0 / array.mean()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size != 1:
        raise ValueError("The matched Table-I protocol requires batch size 1")
    if args.method == "roadformer" and args.precision != "fp32":
        raise ValueError("Pinned MMCV 1.7 deformable attention supports RoadFormer FP32 only")
    if args.warmup <= 0 or args.iterations <= 0 or args.repeats < 2:
        raise ValueError("Use positive warmup/iterations and at least two repeats")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if "RTX 4090 D" not in device_name:
        raise RuntimeError(f"Expected RTX 4090 D, found {device_name}")
    source = args.official_source.resolve() if args.official_source else None
    source_metadata = verify_source(args.method, source)

    operator_bridge = None
    if args.method == "roadformer":
        assert source is not None
        model, forward, operator_bridge = build_roadformer(
            source, args.checkpoint, device, args.height, args.width
        )
    else:
        model, forward = build_standard_model(args.method, source, args.checkpoint, device)

    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    if dtype == torch.float16:
        model.half()
    rgb = torch.randn(
        args.batch_size, 3, args.height, args.width, device=device, dtype=dtype
    )
    geometry_channels = 1 if args.method == "plard" else 3
    geometry = torch.randn(
        args.batch_size,
        geometry_channels,
        args.height,
        args.width,
        device=device,
        dtype=dtype,
    )

    repeat_summaries: list[dict[str, float | int]] = []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            forward(rgb, geometry)
        torch.cuda.synchronize(device)
        for _ in range(args.repeats):
            timings: list[float] = []
            for _ in range(args.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                forward(rgb, geometry)
                end.record()
                end.synchronize()
                timings.append(float(start.elapsed_time(end)))
            repeat_summaries.append(sample_stats(timings))

    repeat_means = [float(item["mean_ms"]) for item in repeat_summaries]
    aggregate_mean = statistics.fmean(repeat_means)
    result = {
        "protocol": "matched Table-I model-only CUDA-event latency",
        "method": args.method,
        "input_shape_rgb": [args.batch_size, 3, args.height, args.width],
        "input_shape_geometry": [
            args.batch_size,
            geometry_channels,
            args.height,
            args.width,
        ],
        "precision": args.precision,
        "backend": "PyTorch",
        "preprocessing_included": False,
        "host_to_device_transfer_included": False,
        "postprocessing_included": False,
        "warmup": args.warmup,
        "iterations_per_repeat": args.iterations,
        "repeats": args.repeats,
        "repeat_summaries": repeat_summaries,
        "aggregate": {
            "mean_ms": aggregate_mean,
            "sample_std_ms_across_repeat_means": statistics.stdev(repeat_means),
            "fps_from_mean": 1000.0 / aggregate_mean,
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_cuda_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "official_source": source_metadata,
        "operator_bridge": operator_bridge,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": device_name,
            "command": sys.argv,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
