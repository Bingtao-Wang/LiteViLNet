#!/usr/bin/env python3
"""Prepare released ORFD splits for RoadFormer's official ORFD loader."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.cache_official_orfd_normals import validate_formal_calibration_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normal-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_symlink(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"Conflicting symlink: {destination}")
        return
    if destination.exists():
        raise FileExistsError(destination)
    destination.symlink_to(source.resolve())


def encode_normal(source: Path, destination: Path) -> None:
    normal = np.load(source, allow_pickle=False)
    if normal.shape[0] != 3 or normal.dtype != np.float32 or not np.isfinite(normal).all():
        raise ValueError(f"Invalid float32 SNE cache: {source}, {normal.shape}, {normal.dtype}")
    encoded = np.rint(
        np.clip((normal.transpose(1, 2, 0) + 1.0) * 0.5, 0.0, 1.0) * 65535.0
    ).astype(np.uint16)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        existing = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
        if existing is not None and np.array_equal(existing, encoded):
            return
        raise FileExistsError(f"Conflicting encoded normal: {destination}")
    if not cv2.imwrite(str(destination), encoded):
        raise RuntimeError(f"Failed to write {destination}")


def prepare_one(task: tuple[str, str, str, str, str, str]) -> None:
    (
        image_source,
        image_destination,
        label_source,
        label_destination,
        normal_source,
        normal_destination,
    ) = task
    ensure_symlink(Path(image_source), Path(image_destination))
    ensure_symlink(Path(label_source), Path(label_destination))
    encode_normal(Path(normal_source), Path(normal_destination))


def main() -> None:
    args = parse_args()
    source_metadata_path = args.normal_root / "normal_cache_metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("profile") != "sne_roadseg":
        raise ValueError("RoadFormer ORFD preparation requires the SNE-RoadSeg normal profile")
    validate_formal_calibration_metadata(source_metadata)
    counts: dict[str, int] = {}
    for split in ("training", "validation", "testing"):
        images = sorted((args.data_root / split / "image_data").glob("*.png"))
        if not images:
            raise FileNotFoundError(args.data_root / split / "image_data")
        counts[split] = len(images)
        tasks = []
        for image in images:
            stem = image.stem
            tasks.append(
                (
                    str(image),
                    str(args.output_root / "images" / split / image.name),
                    str(args.data_root / split / "gt_image" / f"{stem}_fillcolor.png"),
                    str(args.output_root / "annotations" / split / f"{stem}_fillcolor.png"),
                    str(args.normal_root / split / "normal" / f"{stem}.npy"),
                    str(args.output_root / "normal" / split / f"{stem}.png"),
                )
            )
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, _ in enumerate(executor.map(prepare_one, tasks, chunksize=4), start=1):
                if index % 100 == 0 or index == len(images):
                    print(
                        f"prepared RoadFormer ORFD {split}: {index}/{len(images)}",
                        flush=True,
                    )

    metadata = {
        "protocol": "released ORFD train/validation/testing partitions",
        "counts": counts,
        "normal_source": "official SNE-RoadSeg float32 cache",
        "normal_source_metadata_sha256": sha256(source_metadata_path),
        "normal_source_metadata": source_metadata,
        "normal_encoding": "round(clip((normal + 1) / 2, 0, 1) * 65535) as uint16 PNG",
        "image_and_annotation_storage": "symlinks to the released ORFD files",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "matched_split_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
