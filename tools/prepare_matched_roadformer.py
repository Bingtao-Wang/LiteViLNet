#!/usr/bin/env python3
"""Prepare a split-safe KITTI tree for the official RoadFormer loader."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from tools.prepare_matched_kitti_baselines import read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-root", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
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
    """Encode official SNE float normals as RoadFormer's uint16 PNG input."""

    normal = np.load(source, allow_pickle=False)
    if normal.ndim != 3 or normal.shape[0] != 3 or normal.dtype != np.float32:
        raise ValueError(f"Unexpected normal cache format: {source}, {normal.shape}, {normal.dtype}")
    encoded = np.rint(np.clip((normal.transpose(1, 2, 0) + 1.0) * 0.5, 0.0, 1.0) * 65535.0)
    encoded = encoded.astype(np.uint16)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        existing = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
        if existing is not None and np.array_equal(existing, encoded):
            return
        raise FileExistsError(f"Conflicting encoded normal: {destination}")
    if not cv2.imwrite(str(destination), encoded):
        raise RuntimeError(f"Failed to write {destination}")


def prepare_split(root: Path, output: Path, split: str, samples: list[str]) -> None:
    for sample in samples:
        category, frame = sample.split("_", 1)
        ensure_symlink(
            root / split / "image_2" / f"{sample}.png",
            output / "image_2" / split / f"{sample}.png",
        )
        ensure_symlink(
            root / split / "gt_image_2" / f"{category}_road_{frame}.png",
            output / "gt_image_2" / split / f"{sample}.png",
        )
        encode_normal(
            root / split / "normal" / f"{sample}.npy",
            output / "sne" / split / f"{sample}.png",
        )


def main() -> None:
    args = parse_args()
    train = read_manifest(args.train_file)
    val = read_manifest(args.val_file)
    overlap = sorted(set(train) & set(val))
    if overlap:
        raise ValueError(f"Train/validation overlap: {overlap[:5]}")
    prepare_split(args.matched_root, args.output_root, "training", train)
    prepare_split(args.matched_root, args.output_root, "validation", val)
    metadata = {
        "protocol": "matched local KITTI perspective-view retraining",
        "train_count": len(train),
        "val_count": len(val),
        "train_val_overlap": 0,
        "train_manifest_sha256": sha256(args.train_file),
        "val_manifest_sha256": sha256(args.val_file),
        "normal_source": "official SNE-RoadSeg float32 cache",
        "normal_encoding": "round(clip((normal + 1) / 2, 0, 1) * 65535) as uint16 PNG",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "matched_split_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
