#!/usr/bin/env python
"""Benchmark the truthful RGB-D robot path, including depth3 preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from litevilnet.data.robot_road_dataset import load_depth3_tensor, load_rgb_tensor
from litevilnet.models.litevilnet_rgbdepth import LiteViLNetRGBDepth
from litevilnet.utils.common import summarize_ms, system_metadata, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGB-D depth3 end-to-end latency benchmark")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img_h", type=int, default=800)
    parser.add_argument("--img_w", type=int, default=1280)
    parser.add_argument("--max_depth_mm", type=float, default=12000.0)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--output", default="runs/revision_1/robot_depth3_end_to_end.json")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nvidia_processes() -> str:
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except FileNotFoundError:
        return ""


def load_model(checkpoint_path: Path, device: torch.device, dtype: torch.dtype) -> LiteViLNetRGBDepth:
    model = LiteViLNetRGBDepth(pretrained=False, use_deep_supervision=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    bad_missing = [key for key in incompatible.missing_keys if not key.startswith("decoder.aux_heads.")]
    bad_unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("decoder.aux_heads.")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={bad_missing[:8]}, unexpected={bad_unexpected[:8]}")
    model.eval().to(device)
    if dtype == torch.float16:
        model.half()
    return model


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    data_root = Path(args.data_root)
    session_root = data_root / args.session
    if not session_root.exists():
        session_root = data_root / "raw" / args.session
    pairs = []
    for rgb_path in sorted((session_root / "rgb").glob("*.png")):
        depth_path = session_root / "depth" / rgb_path.name
        if depth_path.exists():
            pairs.append((rgb_path, depth_path))
    if not pairs:
        raise FileNotFoundError(f"No paired RGB/depth PNGs below {session_root}")

    checkpoint_path = Path(args.checkpoint)
    model = load_model(checkpoint_path, device, dtype)
    stages = {name: [] for name in ("decode_and_depth3", "host_to_device", "model", "postprocess", "end_to_end")}

    def run_one(index: int, record: bool) -> None:
        rgb_path, depth_path = pairs[index % len(pairs)]
        end_to_end_start = time.perf_counter()
        start = time.perf_counter()
        rgb, original_size = load_rgb_tensor(rgb_path, args.img_h, args.img_w)
        depth3 = load_depth3_tensor(depth_path, args.img_h, args.img_w, args.max_depth_mm)
        rgb = rgb.unsqueeze(0).pin_memory()
        depth3 = depth3.unsqueeze(0).pin_memory()
        decode_ms = (time.perf_counter() - start) * 1000.0

        transfer_start = torch.cuda.Event(enable_timing=True)
        transfer_end = torch.cuda.Event(enable_timing=True)
        transfer_start.record()
        rgb_gpu = rgb.to(device=device, dtype=dtype, non_blocking=True)
        depth_gpu = depth3.to(device=device, dtype=dtype, non_blocking=True)
        transfer_end.record()
        transfer_end.synchronize()
        transfer_ms = float(transfer_start.elapsed_time(transfer_end))

        model_start = torch.cuda.Event(enable_timing=True)
        model_end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            model_start.record()
            logits = model(rgb_gpu, depth_gpu)
            if isinstance(logits, tuple):
                logits = logits[0]
            model_end.record()
            model_end.synchronize()
        model_ms = float(model_start.elapsed_time(model_end))

        post_start = time.perf_counter()
        width, height = original_size
        with torch.inference_mode():
            probabilities = torch.sigmoid(logits.float())
            probabilities = F.interpolate(
                probabilities,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            prediction = (probabilities > 0.5).to(torch.uint8).cpu().numpy()
        torch.cuda.synchronize()
        post_ms = (time.perf_counter() - post_start) * 1000.0
        end_to_end_ms = (time.perf_counter() - end_to_end_start) * 1000.0
        if prediction.shape[0] != 1:
            raise RuntimeError("Unexpected output batch")
        if record:
            stages["decode_and_depth3"].append(decode_ms)
            stages["host_to_device"].append(transfer_ms)
            stages["model"].append(model_ms)
            stages["postprocess"].append(post_ms)
            stages["end_to_end"].append(end_to_end_ms)

    for index in range(args.warmup):
        run_one(index, record=False)
    process_snapshot_before = nvidia_processes()
    torch.cuda.reset_peak_memory_stats()
    for index in range(args.iters):
        run_one(args.warmup + index, record=True)

    payload = {
        "protocol": {
            "pipeline": "PNG decode + RGB normalization + aligned-depth depth3 encoding + pinned H2D + model + sigmoid/resize/threshold/D2H",
            "geometry_representation": "depth3=[clipped normalized depth, valid mask, inverse depth], mapped to [-1,1]; not ADI",
            "input_shape_per_modality": [1, 3, args.img_h, args.img_w],
            "precision": args.precision,
            "batch_size": 1,
            "warmup": args.warmup,
            "iterations": args.iters,
            "disk_note": "real PNG files; warm-up makes later reads eligible for the operating-system page cache",
        },
        "data_root": str(data_root),
        "session": args.session,
        "paired_frames": len(pairs),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "timing": {name: summarize_ms(values) for name, values in stages.items()},
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "system": system_metadata(),
        "nvidia_compute_processes_before": process_snapshot_before,
        "nvidia_compute_processes_after": nvidia_processes(),
        "power_measurement": None,
        "navigation_metrics": None,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
