#!/usr/bin/env python
"""Fair PyTorch latency benchmark for VLLiNet presets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from litevilnet.utils.common import append_csv, summarize_ms, system_metadata, write_json
from litevilnet.model_factory import MODEL_PRESETS, available_presets, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LiteViLNet presets in PyTorch")
    parser.add_argument("--preset", default="vllinet_paper", choices=available_presets())
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    parser.add_argument("--img_h", type=int, default=384)
    parser.add_argument("--img_w", type=int, default=1248)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--measure_preprocess", action="store_true")
    parser.add_argument("--rgb_image", default="")
    parser.add_argument("--adi_image", default="")
    parser.add_argument("--output", default="deployment/results/pytorch_benchmark.json")
    parser.add_argument("--csv", default="deployment/results/benchmark_summary.csv")
    parser.add_argument("--allow_partial", action="store_true", help="Allow non-aux partial checkpoint loading")
    return parser.parse_args()


def _load_or_random_image(path: str, h: int, w: int) -> np.ndarray:
    if path:
        image = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0
    return np.random.rand(h, w, 3).astype(np.float32)


def preprocess_pair(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = _load_or_random_image(args.rgb_image, args.img_h, args.img_w)
    adi = _load_or_random_image(args.adi_image, args.img_h, args.img_w)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = ((rgb - mean) / std).transpose(2, 0, 1)
    adi = ((adi - mean) / std).transpose(2, 0, 1)
    rgb_tensor = torch.from_numpy(rgb).unsqueeze(0).repeat(args.batch_size, 1, 1, 1).to(device=device, dtype=dtype)
    adi_tensor = torch.from_numpy(adi).unsqueeze(0).repeat(args.batch_size, 1, 1, 1).to(device=device, dtype=dtype)
    return rgb_tensor, adi_tensor


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for deployment latency benchmark")

    device = torch.device("cuda")
    checkpoint = args.checkpoint or MODEL_PRESETS[args.preset].get("checkpoint_hint", "")
    checkpoint = checkpoint if checkpoint and Path(checkpoint).exists() else None
    model, metadata = build_model(
        args.preset,
        checkpoint=checkpoint,
        device=device,
        strict=False,
        allow_partial=args.allow_partial,
    )
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    if args.precision == "fp16":
        model.half()

    rgb = torch.randn(args.batch_size, 3, args.img_h, args.img_w, device=device, dtype=dtype)
    adi = torch.randn(args.batch_size, 3, args.img_h, args.img_w, device=device, dtype=dtype)

    preprocess_times = []
    if args.measure_preprocess:
        for _ in range(args.iters):
            start = time.perf_counter()
            rgb, adi = preprocess_pair(args, device, dtype)
            torch.cuda.synchronize()
            preprocess_times.append((time.perf_counter() - start) * 1000.0)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(rgb, adi)
    torch.cuda.synchronize()

    model_times = []
    with torch.no_grad():
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(rgb, adi)
            end.record()
            torch.cuda.synchronize()
            model_times.append(start.elapsed_time(end))

    end_to_end_times = []
    if args.measure_preprocess:
        with torch.no_grad():
            for _ in range(args.iters):
                wall_start = time.perf_counter()
                rgb, adi = preprocess_pair(args, device, dtype)
                logits = model(rgb, adi)
                _ = (torch.sigmoid(logits) > 0.5).to(torch.uint8)
                torch.cuda.synchronize()
                end_to_end_times.append((time.perf_counter() - wall_start) * 1000.0)

    result = {
        "backend": "pytorch",
        "preset": args.preset,
        "label": metadata["label"],
        "precision": args.precision,
        "checkpoint": checkpoint,
        "input_shape": [args.batch_size, 3, args.img_h, args.img_w],
        "warmup": args.warmup,
        "iters": args.iters,
        "parameters": metadata["parameters"],
        "model_only": summarize_ms(model_times),
        "preprocess": summarize_ms(preprocess_times),
        "end_to_end": summarize_ms(end_to_end_times),
        "cuda_max_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "system": system_metadata(),
    }
    write_json(args.output, result)

    row = {
        "backend": "pytorch",
        "model": metadata["label"],
        "preset": args.preset,
        "precision": args.precision,
        "batch_size": args.batch_size,
        "img_h": args.img_h,
        "img_w": args.img_w,
        "mean_ms": result["model_only"]["mean_ms"],
        "p95_ms": result["model_only"]["p95_ms"],
        "fps": result["model_only"]["fps"] * args.batch_size,
        "parameters_M": metadata["parameters"] / 1e6,
        "memory_mb": result["cuda_max_memory_mb"],
    }
    append_csv(args.csv, row)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
