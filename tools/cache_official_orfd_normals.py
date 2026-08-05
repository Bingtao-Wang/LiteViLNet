#!/usr/bin/env python3
"""Cache ORFD surface normals from a pinned authors' SNE implementation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
import torch


_SNE = None
_DEVICE = torch.device("cpu")
ORFD_SAMPLE_COUNT = 11_830


def validate_formal_calibration_metadata(metadata: dict[str, object]) -> None:
    """Require the complete released one-calibration-per-sample archive."""

    calibration = metadata.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("ORFD normal metadata omits calibration provenance")
    checks = {
        "released_calibration_files": (
            int(calibration.get("released_calibration_files", -1)),
            ORFD_SAMPLE_COUNT,
        ),
        "exact_sample_matches": (
            int(calibration.get("exact_sample_matches", -1)),
            ORFD_SAMPLE_COUNT,
        ),
        "nearest_timestamp_matches": (
            int(calibration.get("nearest_timestamp_matches", -1)),
            0,
        ),
        "maximum_nearest_timestamp_gap": (
            int(calibration.get("maximum_nearest_timestamp_gap", -1)),
            0,
        ),
        "inferred_intrinsic_counts": (
            calibration.get("inferred_intrinsic_counts"),
            {},
        ),
    }
    mismatches = {key: values for key, values in checks.items() if values[0] != values[1]}
    if mismatches:
        raise ValueError(
            "Formal ORFD runs require the complete released per-frame calibration set: "
            f"{mismatches}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("sne_roadseg", "offnet"),
        required=True,
        help="Select the pinned SNE file and its released ORFD preprocessing",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Normal-generation device; use one worker for a CUDA device",
    )
    parser.add_argument("--include-flipped-training", action="store_true")
    parser.add_argument(
        "--require-exact-calibration",
        action="store_true",
        help=(
            "Reject an incomplete extraction unless every dense-depth timestamp "
            "has its released calibration file"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(path: Path, arguments: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_cam_k(path: Path) -> np.ndarray:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("cam_K:"):
            values = np.asarray(line.split(":", 1)[1].split(), dtype=np.float32)
            return values.reshape(3, 3)
    raise ValueError(f"cam_K not found in {path}")


def calibration_index(root: Path) -> tuple[list[int], dict[int, tuple[Path, np.ndarray]]]:
    records: dict[int, tuple[Path, np.ndarray]] = {}
    for split in ("training", "validation", "testing"):
        for path in sorted((root / split / "calib").glob("*.txt")):
            timestamp = int(path.stem)
            camera = read_cam_k(path)
            if timestamp in records and not np.array_equal(records[timestamp][1], camera):
                raise ValueError(f"Conflicting calibration for timestamp {timestamp}")
            records[timestamp] = (path, camera)
    if not records:
        raise FileNotFoundError(f"No ORFD calib files below {root}")
    return sorted(records), records


def nearest_calibration(
    timestamp: int, timestamps: list[int], records: dict[int, tuple[Path, np.ndarray]]
) -> tuple[Path, np.ndarray, int, bool]:
    if timestamp in records:
        path, camera = records[timestamp]
        return path, camera.copy(), 0, True
    index = bisect_left(timestamps, timestamp)
    candidates = timestamps[max(0, index - 1) : min(len(timestamps), index + 1)]
    nearest = min(candidates, key=lambda value: abs(value - timestamp))
    path, camera = records[nearest]
    return path, camera.copy(), abs(nearest - timestamp), False


def worker_init(official_source: str, device_string: str = "cpu") -> None:
    global _SNE, _DEVICE
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _DEVICE = torch.device(device_string)
    if _DEVICE.type == "cuda":
        # The pinned authors' SNE creates tensors without explicit devices.
        # A per-process default keeps those tensors colocated with the input.
        if hasattr(torch, "set_default_device"):
            torch.set_default_device(_DEVICE)
        else:
            # PyTorch 1.x compatibility for the pinned RoadFormer environment.
            torch.cuda.set_device(_DEVICE)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
    source = str(Path(official_source).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from models.sne_model import SNE  # pylint: disable=import-outside-toplevel

    try:
        _SNE = SNE(crop_top=True).eval()
    except TypeError:
        _SNE = SNE().eval()


def compute_one(task: tuple[str, str, list[list[float]], bool, bool, float]) -> dict[str, object]:
    depth_string, output_string, camera_values, flipped, force, depth_divisor = task
    depth_path = Path(depth_string)
    output_path = Path(output_string)
    if output_path.is_file() and not force:
        cached = np.load(output_path, mmap_mode="r", allow_pickle=False)
        if cached.shape != (3, 720, 1280) or cached.dtype != np.float32:
            raise ValueError(
                f"Invalid existing normal cache: {output_path}, "
                f"{cached.shape}, {cached.dtype}"
            )
        return {"path": output_string, "cached": True}

    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"Expected single-channel uint16 depth PNG: {depth_path}")
    depth_float = depth.astype(np.float32) / depth_divisor
    if flipped:
        depth_float = np.ascontiguousarray(depth_float[:, ::-1])
    camera = torch.tensor(camera_values, dtype=torch.float32, device=_DEVICE)
    # The released OFF-Net ORFD loader applies this 720 -> 704 vertical offset
    # before SNE. The same registered ORFD geometry is used for all profiles.
    camera[1, 2] -= 8.0
    with torch.no_grad():
        depth_tensor = torch.from_numpy(depth_float).to(_DEVICE)
        normal = _SNE(depth_tensor, camera).cpu().numpy().astype(np.float32)
    if normal.shape != (3, depth.shape[0], depth.shape[1]) or not np.isfinite(normal).all():
        raise ValueError(f"Invalid normal for {depth_path}: {normal.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        np.save(handle, normal, allow_pickle=False)
    os.replace(temporary_path, output_path)
    return {"path": output_string, "cached": False}


def collect_tasks(args: argparse.Namespace) -> tuple[list[tuple], dict[str, object]]:
    timestamps, records = calibration_index(args.data_root)
    tasks: list[tuple] = []
    exact = 0
    inferred = 0
    maximum_gap = 0
    inferred_calibrations: dict[str, int] = {}
    depth_divisor = 256.0 if args.profile == "offnet" else 1000.0
    for split in ("training", "validation", "testing"):
        depth_paths = sorted((args.data_root / split / "dense_depth").glob("*.png"))
        if not depth_paths:
            raise FileNotFoundError(args.data_root / split / "dense_depth")
        for depth_path in depth_paths:
            calib_path, camera, gap, is_exact = nearest_calibration(
                int(depth_path.stem), timestamps, records
            )
            del calib_path
            exact += int(is_exact)
            inferred += int(not is_exact)
            maximum_gap = max(maximum_gap, gap)
            if not is_exact:
                key = ",".join(f"{value:.6f}" for value in camera.reshape(-1))
                inferred_calibrations[key] = inferred_calibrations.get(key, 0) + 1
            output = args.output_root / split / "normal" / f"{depth_path.stem}.npy"
            tasks.append(
                (
                    str(depth_path),
                    str(output),
                    camera.tolist(),
                    False,
                    args.force,
                    depth_divisor,
                )
            )
            if split == "training" and args.include_flipped_training:
                flipped_output = args.output_root / split / "normal_flipped" / f"{depth_path.stem}.npy"
                tasks.append(
                    (
                        str(depth_path),
                        str(flipped_output),
                        camera.tolist(),
                        True,
                        args.force,
                        depth_divisor,
                    )
                )
    calibration_metadata = {
        "released_calibration_files": len(records),
        "exact_sample_matches": exact,
        "nearest_timestamp_matches": inferred,
        "maximum_nearest_timestamp_gap": maximum_gap,
        "inferred_intrinsic_counts": inferred_calibrations,
        "matching_rule": "exact timestamp when available, otherwise nearest released ORFD timestamp",
    }
    if args.require_exact_calibration and inferred:
        raise ValueError(
            "Exact ORFD calibration was required, but "
            f"{inferred} of {exact + inferred} samples have no timestamp-matched "
            "calibration file. Re-extract the complete released archive."
        )
    return tasks, calibration_metadata


def main() -> None:
    args = parse_args()
    if torch.device(args.device).type == "cuda" and args.workers != 1:
        raise ValueError("CUDA normal generation requires --workers 1")
    expected_files = {
        "sne_roadseg": args.official_source / "models" / "sne_model.py",
        "offnet": args.official_source / "models" / "sne_model.py",
    }
    official_file = expected_files[args.profile]
    if not official_file.is_file():
        raise FileNotFoundError(official_file)
    if git_value(args.official_source, ["status", "--porcelain"]):
        raise RuntimeError(f"Official source must be clean: {args.official_source}")
    tasks, calibration_metadata = collect_tasks(args)
    if args.require_exact_calibration:
        validate_formal_calibration_metadata({"calibration": calibration_metadata})
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    completed = 0
    reused = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(str(args.official_source), args.device),
    ) as executor:
        for result in executor.map(compute_one, tasks, chunksize=4):
            completed += 1
            reused += int(bool(result["cached"]))
            if completed % 100 == 0 or completed == len(tasks):
                print(f"cached ORFD normals: {completed}/{len(tasks)} (reused {reused})", flush=True)

    metadata = {
        "dataset": "ORFD",
        "profile": args.profile,
        "generator": str(official_file.relative_to(args.official_source)),
        "official_source": str(args.official_source.resolve()),
        "official_remote": git_value(args.official_source, ["remote", "get-url", "origin"]),
        "official_commit": git_value(args.official_source, ["rev-parse", "HEAD"]),
        "official_sne_sha256": sha256(official_file),
        "storage_dtype": "float32",
        "generation_device": args.device,
        "depth_divisor": 256 if args.profile == "offnet" else 1000,
        "camera_vertical_offset_pixels": -8,
        "task_count": len(tasks),
        "reused_count": reused,
        "training_flip_semantics": (
            "depth is flipped before official SNE"
            if args.include_flipped_training
            else "no flipped cache requested"
        ),
        "calibration": calibration_metadata,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "normal_cache_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
