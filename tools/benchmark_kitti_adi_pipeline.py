#!/usr/bin/env python
"""Benchmark real KITTI LiDAR-to-ADI preprocessing and PyTorch inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from litevilnet.data.adi import generate_adi, read_kitti_calibration
from litevilnet.models.vllinet import VLLiNet_Lite
from litevilnet.utils.common import summarize_ms, system_metadata, write_json


RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KITTI raw-LiDAR ADI end-to-end benchmark")
    parser.add_argument("--data_root", required=True, help="KITTI Road root with image_2/calib/ADI")
    parser.add_argument("--velodyne_root", required=True, help="Extracted data_road_velodyne root")
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img_h", type=int, default=384)
    parser.add_argument("--img_w", type=int, default=1248)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=58)
    parser.add_argument("--output", default="runs/revision_1/kitti_adi_end_to_end.json")
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


def find_velodyne(root: Path, sample_id: str) -> Path:
    candidates = [
        root / "training" / "velodyne" / f"{sample_id}.bin",
        root / "data_road_velodyne" / "training" / "velodyne" / f"{sample_id}.bin",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Velodyne file unavailable for {sample_id}: {candidates}")


def load_model(path: Path, device: torch.device, dtype: torch.dtype) -> VLLiNet_Lite:
    model = VLLiNet_Lite(pretrained=False, use_deep_supervision=True)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
        raise SystemExit("CUDA is required for the full pipeline benchmark")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    data_root = Path(args.data_root)
    velodyne_root = Path(args.velodyne_root)
    sample_ids = [
        line.strip().removesuffix(".png")
        for line in Path(args.split_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not sample_ids:
        raise ValueError("Empty split manifest")
    checkpoint_path = Path(args.checkpoint)
    model = load_model(checkpoint_path, device, dtype)
    stage_names = (
        "disk_and_calibration",
        "projection",
        "interpolation",
        "gradient_and_normalization",
        "tensor_preparation",
        "cpu_preprocess_total",
        "host_to_device",
        "model",
        "postprocess",
        "end_to_end",
    )
    stages = {name: [] for name in stage_names}
    reference_mae = []

    def run_one(index: int, record: bool) -> None:
        sample_id = sample_ids[index % len(sample_ids)]
        start_end_to_end = time.perf_counter()
        start = time.perf_counter()
        rgb_bgr = cv2.imread(str(data_root / "training" / "image_2" / f"{sample_id}.png"))
        if rgb_bgr is None:
            raise FileNotFoundError(sample_id)
        original_height, original_width = rgb_bgr.shape[:2]
        projection = read_kitti_calibration(data_root / "training" / "calib" / f"{sample_id}.txt")
        points = np.fromfile(find_velodyne(velodyne_root, sample_id), dtype=np.float32).reshape(-1, 4)
        load_ms = (time.perf_counter() - start) * 1000.0

        adi, adi_times = generate_adi(
            points,
            projection,
            original_height,
            original_width,
            return_timings=True,
        )
        start = time.perf_counter()
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (args.img_w, args.img_h), interpolation=cv2.INTER_LINEAR)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = ((rgb - RGB_MEAN) / RGB_STD).transpose(2, 0, 1)
        adi_resized = cv2.resize(adi, (args.img_w, args.img_h), interpolation=cv2.INTER_LINEAR)
        adi3 = np.repeat(adi_resized[:, :, None], 3, axis=2)
        adi3 = ((adi3 - RGB_MEAN) / RGB_STD).transpose(2, 0, 1)
        rgb_tensor = torch.from_numpy(rgb.copy()).unsqueeze(0).pin_memory()
        adi_tensor = torch.from_numpy(adi3.copy()).unsqueeze(0).pin_memory()
        tensor_ms = (time.perf_counter() - start) * 1000.0

        transfer_start = torch.cuda.Event(enable_timing=True)
        transfer_end = torch.cuda.Event(enable_timing=True)
        transfer_start.record()
        rgb_gpu = rgb_tensor.to(device=device, dtype=dtype, non_blocking=True)
        adi_gpu = adi_tensor.to(device=device, dtype=dtype, non_blocking=True)
        transfer_end.record()
        transfer_end.synchronize()
        transfer_ms = float(transfer_start.elapsed_time(transfer_end))

        model_start = torch.cuda.Event(enable_timing=True)
        model_end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            model_start.record()
            logits = model(rgb_gpu, adi_gpu)
            if isinstance(logits, tuple):
                logits = logits[0]
            model_end.record()
            model_end.synchronize()
        model_ms = float(model_start.elapsed_time(model_end))

        post_start = time.perf_counter()
        with torch.inference_mode():
            probabilities = F.interpolate(
                torch.sigmoid(logits.float()),
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
            prediction = (probabilities > 0.5).to(torch.uint8).cpu().numpy()
        torch.cuda.synchronize()
        post_ms = (time.perf_counter() - post_start) * 1000.0
        end_to_end_ms = (time.perf_counter() - start_end_to_end) * 1000.0
        if prediction.shape[0] != 1:
            raise RuntimeError("Unexpected prediction batch")

        reference = cv2.imread(
            str(data_root / "training" / "ADI" / f"{sample_id}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if record and reference is not None and reference.shape == adi.shape:
            reference_mae.append(float(np.mean(np.abs(reference.astype(np.float32) / 255.0 - adi))))
        if record:
            stages["disk_and_calibration"].append(load_ms)
            stages["projection"].append(adi_times["projection_ms"])
            stages["interpolation"].append(adi_times["interpolation_ms"])
            stages["gradient_and_normalization"].append(
                adi_times["gradient_and_normalization_ms"]
            )
            stages["tensor_preparation"].append(tensor_ms)
            stages["cpu_preprocess_total"].append(load_ms + adi_times["adi_total_ms"] + tensor_ms)
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
            "pipeline": "real PNG/BIN/calibration load + PLARD ADI + tensor normalization + pinned H2D + PyTorch model + sigmoid/resize/threshold/D2H",
            "adi": "nearest duplicate projection; 21x21 inverse-distance interpolation; 7x7 altitude-gradient average; per-image sqrt/Gaussian/clip normalization",
            "input_shape_per_modality": [1, 3, args.img_h, args.img_w],
            "precision": args.precision,
            "batch_size": 1,
            "warmup": args.warmup,
            "iterations": args.iters,
            "split_file": args.split_file,
            "disk_note": "real files; timed samples follow warm-up and may use the operating-system page cache",
        },
        "data_root": str(data_root),
        "velodyne_root": str(velodyne_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "timing": {name: summarize_ms(values) for name, values in stages.items()},
        "generated_vs_stored_adi_mae": {
            "count": len(reference_mae),
            "mean": float(np.mean(reference_mae)) if reference_mae else None,
            "sample_std": float(np.std(reference_mae, ddof=1)) if len(reference_mae) > 1 else 0.0,
            "p95": float(np.percentile(reference_mae, 95)) if reference_mae else None,
            "note": "diagnostic only; stored ADIs may reflect release-code indexing/encoding details",
        },
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "system": system_metadata(),
        "nvidia_compute_processes_before": process_snapshot_before,
        "nvidia_compute_processes_after": nvidia_processes(),
        "power_measurement": None,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
