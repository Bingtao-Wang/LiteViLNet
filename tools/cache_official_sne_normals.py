#!/usr/bin/env python3
"""Cache SNE surface normals using the official SNE-RoadSeg implementation.

Normal estimation is deterministic but expensive when repeated in every data
loader epoch.  This script imports the official ``models/sne_model.py``
unchanged, evaluates it on CPU, and stores float32 arrays.  Both original and
horizontally flipped training depths are cached because the official USNet
augmentation applies the flip before SNE.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


_SNE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("sne_roadseg", "offnet"),
        default="sne_roadseg",
        help="Record and verify which authors' SNE implementation is imported",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-flipped-val", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_p2(path: Path) -> np.ndarray:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("P2:"):
            return np.asarray([float(value) for value in line.split(":", 1)[1].split()], dtype=np.float32).reshape(3, 4)
    raise ValueError(f"P2 not found in {path}")


def worker_init(official_source: str) -> None:
    global _SNE
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = str(Path(official_source).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from models.sne_model import SNE  # pylint: disable=import-outside-toplevel

    try:
        _SNE = SNE(crop_top=True).eval()
    except TypeError:
        # SNE-RoadSeg always enables the top crop and therefore exposes no
        # constructor flag; USNet vendors the same implementation with an
        # explicit ``crop_top=True`` argument.
        _SNE = SNE().eval()


def compute_one(task: tuple[str, str, str, bool, bool]) -> dict[str, object]:
    depth_path_string, calib_path_string, output_path_string, flipped, force = task
    depth_path = Path(depth_path_string)
    calib_path = Path(calib_path_string)
    output_path = Path(output_path_string)
    if output_path.is_file() and not force:
        cached = np.load(output_path, mmap_mode="r", allow_pickle=False)
        if cached.shape != (3, 384, 1248) or cached.dtype != np.float32:
            raise ValueError(
                f"Invalid existing normal cache: {output_path}, "
                f"{cached.shape}, {cached.dtype}"
            )
        return {"path": output_path_string, "cached": True}

    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if depth is None or depth.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth PNG: {depth_path}")
    depth_m = depth.astype(np.float32) / 1000.0
    if flipped:
        depth_m = np.ascontiguousarray(depth_m[:, ::-1])
    cam = torch.from_numpy(read_p2(calib_path))
    with torch.no_grad():
        normal = _SNE(torch.from_numpy(depth_m), cam).cpu().numpy().astype(np.float32, copy=False)
    if normal.shape != (3, depth.shape[0], depth.shape[1]) or not np.isfinite(normal).all():
        raise ValueError(f"Invalid normal array for {depth_path}: {normal.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        np.save(handle, normal, allow_pickle=False)
    os.replace(temporary_path, output_path)
    return {"path": output_path_string, "cached": False}


def collect_tasks(args: argparse.Namespace) -> list[tuple[str, str, str, bool, bool]]:
    tasks: list[tuple[str, str, str, bool, bool]] = []
    for split in ("training", "validation"):
        depth_dir = args.data_root / split / "depth_u16"
        calib_dir = args.data_root / split / "calib"
        for depth_path in sorted(depth_dir.glob("*.png")):
            sample = depth_path.stem
            calib_path = calib_dir / f"{sample}.txt"
            if not calib_path.is_file():
                raise FileNotFoundError(calib_path)
            tasks.append(
                (str(depth_path), str(calib_path), str(args.output_root / split / "normal" / f"{sample}.npy"), False, args.force)
            )
            needs_flipped = args.profile == "sne_roadseg" and (
                split == "training" or args.include_flipped_val
            )
            if needs_flipped:
                tasks.append(
                    (str(depth_path), str(calib_path), str(args.output_root / split / "normal_flipped" / f"{sample}.npy"), True, args.force)
                )
    return tasks


def main() -> None:
    args = parse_args()
    official_file = args.official_source / "models" / "sne_model.py"
    if not official_file.is_file():
        raise FileNotFoundError(official_file)
    tasks = collect_tasks(args)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    completed = 0
    reused = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(str(args.official_source),),
    ) as executor:
        for result in executor.map(compute_one, tasks):
            completed += 1
            reused += int(bool(result["cached"]))
            if completed % 25 == 0 or completed == len(tasks):
                print(f"cached normals: {completed}/{len(tasks)} (reused {reused})", flush=True)

    expected = {
        "sne_roadseg": (
            "https://github.com/hlwang1124/SNE-RoadSeg.git",
            "5e7900bfd59887634ced687ffe85a73018a38659",
        ),
        "offnet": (
            "https://github.com/chaytonmin/Off-Road-Freespace-Detection",
            "50e63d24836198e8fb5af707e521f414104b4876",
        ),
    }
    remote = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=args.official_source, text=True
    ).strip()
    commit = git_commit(args.official_source)
    if (remote, commit) != expected[args.profile]:
        raise RuntimeError(
            f"Unexpected {args.profile} source: remote={remote!r}, commit={commit!r}"
        )
    metadata = {
        "generator": f"official {args.profile} models/sne_model.py",
        "profile": args.profile,
        "official_repository": remote,
        "official_source": str(args.official_source.resolve()),
        "official_commit": commit,
        "official_sne_sha256": sha256(official_file),
        "storage_dtype": "float32",
        "units": "depth_u16 / 1000 (metres)",
        "task_count": len(tasks),
        "reused_count": reused,
        "training_flip_semantics": (
            "depth is horizontally flipped before official SNE"
            if args.profile == "sne_roadseg"
            else "not generated for OFF-Net"
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "normal_cache_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
