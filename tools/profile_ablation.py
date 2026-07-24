#!/usr/bin/env python
"""Profile LiteViLNet ablations with reproducible cost and latency metadata.

The FLOP report uses fvcore's JIT analysis and preserves its unsupported-op
list instead of silently presenting a partial count as exact.  CUDA latency is
model-only (random resident tensors); data loading, ADI construction, transfer,
and post-processing are deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from fvcore.nn import FlopCountAnalysis

from litevilnet.models.vllinet_ablation import get_ablation_model
from litevilnet.utils.common import summarize_ms, system_metadata, write_json


DEFAULT_CONFIGS = [
    "baseline",
    "add_lidar",
    "add_fusion",
    "add_bridge",
    "add_deep_sup",
    "optimal",
    "full",
    "transformer_bridge",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile LiteViLNet ablation configurations")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--img_h", type=int, default=384)
    parser.add_argument("--img_w", type=int, default=1248)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--skip_flops",
        action="store_true",
        help="Skip the CPU fvcore analysis (useful for timing-only repeats)",
    )
    parser.add_argument(
        "--skip_timing",
        action="store_true",
        help="Report parameters/FLOPs only; no CUDA timing or memory measurement",
    )
    parser.add_argument("--output", default="runs/revision_1/ablation_profile.json")
    return parser.parse_args()


def _nvidia_process_snapshot() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except FileNotFoundError:
        return []
    rows = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",", 3)]
        if len(values) == 4:
            rows.append(dict(zip(("gpu_uuid", "pid", "process_name", "used_memory_mb"), values)))
    return rows


def _flop_report(config: str, height: int, width: int, batch_size: int) -> dict[str, Any]:
    model = get_ablation_model(config, pretrained=False).eval()
    rgb = torch.randn(batch_size, 3, height, width)
    adi = torch.randn(batch_size, 3, height, width)
    analysis = FlopCountAnalysis(model, (rgb, adi))
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    total = int(analysis.total())
    unsupported = {name: int(count) for name, count in analysis.unsupported_ops().items()}
    by_operator = {name: int(count) for name, count in analysis.by_operator().items()}
    return {
        # fvcore counts one fused multiply-add as one operation.  We therefore
        # call this MAC-equivalent cost, not "exact FLOPs".
        "fvcore_total_mac_equivalents": total,
        "fvcore_total_gmac_equivalents": total / 1e9,
        "fvcore_unsupported_ops": unsupported,
        "fvcore_by_operator": by_operator,
        "inference_auxiliary_heads_executed": False,
    }


def _latency_report(
    config: str,
    height: int,
    width: int,
    batch_size: int,
    precision: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA timing requested but CUDA is unavailable")
    device = torch.device("cuda")
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model = get_ablation_model(config, pretrained=False).eval().to(device)
    if dtype == torch.float16:
        model = model.half()
    rgb = torch.randn(batch_size, 3, height, width, device=device, dtype=dtype)
    adi = torch.randn(batch_size, 3, height, width, device=device, dtype=dtype)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()

    with torch.inference_mode():
        for _ in range(warmup):
            model(rgb, adi)
    torch.cuda.synchronize()
    # Warm-up may create persistent cuDNN workspaces; use the post-warm-up
    # allocation as the baseline for incremental forward-pass memory.
    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    times = []
    with torch.inference_mode():
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(rgb, adi)
            end.record()
            end.synchronize()
            times.append(float(start.elapsed_time(end)))

    peak_allocated = torch.cuda.max_memory_allocated()
    report = {
        "latency_ms": summarize_ms(times),
        "throughput_fps": (1000.0 / (sum(times) / len(times))) * batch_size,
        "cuda_allocated_before_forward_mb": baseline_allocated / (1024**2),
        "cuda_peak_allocated_mb": peak_allocated / (1024**2),
        "cuda_incremental_peak_mb": max(0, peak_allocated - baseline_allocated) / (1024**2),
        "scope": "model-only; random resident tensors; preprocessing, transfer, and post-processing excluded",
    }
    del model, rgb, adi
    torch.cuda.empty_cache()
    return report


def main() -> None:
    args = parse_args()
    unknown = set(args.configs) - set(DEFAULT_CONFIGS)
    if unknown:
        raise SystemExit(f"Unknown configurations: {sorted(unknown)}")
    if args.device == "cpu" and not args.skip_timing:
        raise SystemExit("CUDA event timing requires --device cuda; use --skip_timing for CPU cost analysis")

    torch.backends.cudnn.benchmark = True
    process_snapshot_before = _nvidia_process_snapshot()
    results = []
    for config in args.configs:
        parameter_model = get_ablation_model(config, pretrained=False)
        result: dict[str, Any] = {
            "config": config,
            "parameters": sum(parameter.numel() for parameter in parameter_model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in parameter_model.parameters() if parameter.requires_grad
            ),
        }
        del parameter_model
        if not args.skip_flops:
            result["cost"] = _flop_report(config, args.img_h, args.img_w, args.batch_size)
        if not args.skip_timing:
            result["cuda"] = _latency_report(
                config,
                args.img_h,
                args.img_w,
                args.batch_size,
                args.precision,
                args.warmup,
                args.iters,
            )
        results.append(result)

    payload = {
        "protocol": {
            "input_shape_per_modality": [args.batch_size, 3, args.img_h, args.img_w],
            "precision": args.precision,
            "warmup": args.warmup,
            "iterations": args.iters,
            "pretrained_weights_loaded": False,
            "deep_supervision_auxiliary_heads_used_during_inference": False,
            "fvcore_convention": "one fused multiply-add is counted as one MAC-equivalent operation",
        },
        "system": system_metadata(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_compute_processes_before": process_snapshot_before,
        "nvidia_compute_processes_after": _nvidia_process_snapshot(),
        "results": results,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
