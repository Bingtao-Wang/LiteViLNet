#!/usr/bin/env python3
"""Prepare one leak-free KITTI split for local baseline retraining.

The public PLARD, USNet, and SNE-RoadSeg repositories expect different
directory names for the same validation subset.  This utility creates a
lightweight symlink tree in which all three loaders see the exact sample IDs
listed in the supplied manifests.  No image data are copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SUBDIRS = ("image_2", "gt_image_2", "calib", "ADI", "depth_u16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[str]:
    entries = [
        line.strip().removesuffix(".png")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(entries) != len(set(entries)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return entries


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(data_root: Path, depth_root: Path, subdir: str, sample: str) -> Path:
    if subdir == "gt_image_2":
        category, frame = sample.split("_", 1)
        return data_root / "training" / subdir / f"{category}_road_{frame}.png"
    if subdir == "depth_u16":
        return depth_root / "training" / f"{sample}.png"
    extension = ".txt" if subdir == "calib" else ".png"
    return data_root / "training" / subdir / f"{sample}{extension}"


def link_split(
    output_root: Path,
    split_name: str,
    samples: list[str],
    data_root: Path,
    depth_root: Path,
    force: bool,
) -> None:
    for subdir in SUBDIRS:
        target_dir = output_root / split_name / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            source = source_path(data_root, depth_root, subdir, sample).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Missing {subdir} input for {sample}: {source}")
            target = target_dir / source.name
            if target.is_symlink() and target.resolve() == source:
                continue
            if target.exists() or target.is_symlink():
                if not force:
                    raise FileExistsError(f"Refusing to replace {target}; pass --force")
                target.unlink()
            target.symlink_to(source)


def main() -> None:
    args = parse_args()
    train_samples = read_manifest(args.train_file)
    val_samples = read_manifest(args.val_file)
    overlap = sorted(set(train_samples) & set(val_samples))
    if overlap:
        raise ValueError(f"Train/validation overlap: {overlap[:5]}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    link_split(
        args.output_root,
        "training",
        train_samples,
        args.data_root,
        args.depth_root,
        args.force,
    )
    link_split(
        args.output_root,
        "validation",
        val_samples,
        args.data_root,
        args.depth_root,
        args.force,
    )

    # USNet calls this split "validating" while SNE-RoadSeg calls it
    # "validation".  Point both names to the same physical directory.
    validating = args.output_root / "validating"
    if validating.is_symlink() and validating.resolve() == (args.output_root / "validation").resolve():
        pass
    else:
        if validating.exists() or validating.is_symlink():
            if not args.force:
                raise FileExistsError(f"Refusing to replace {validating}; pass --force")
            if validating.is_dir() and not validating.is_symlink():
                raise IsADirectoryError(f"Remove the real directory before replacing it: {validating}")
            validating.unlink()
        validating.symlink_to("validation", target_is_directory=True)

    metadata = {
        "protocol": "matched local KITTI perspective-view retraining",
        "data_root": str(args.data_root.resolve()),
        "depth_root": str(args.depth_root.resolve()),
        "train_manifest": str(args.train_file.resolve()),
        "val_manifest": str(args.val_file.resolve()),
        "train_manifest_sha256": file_sha256(args.train_file),
        "val_manifest_sha256": file_sha256(args.val_file),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "train_val_overlap": 0,
        "subdirectories": list(SUBDIRS),
    }
    metadata_path = args.output_root / "matched_split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + os.linesep, encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
