#!/usr/bin/env python3
"""Evaluate the authors' released OFF-Net checkpoint on ORFD testing."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train_matched_kitti_baseline import seed_worker
from tools.cache_official_orfd_normals import validate_formal_calibration_metadata
from tools.train_matched_orfd_baseline import (
    ORFDNormalDataset,
    evaluate,
    git_value,
    load_offnet,
)


PINNED_COMMIT = "50e63d24836198e8fb5af707e521f414104b4876"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normal-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=int, default=30)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if git_value(args.official_source, ["rev-parse", "HEAD"]) != PINNED_COMMIT:
        raise RuntimeError(f"OFF-Net source must be pinned at {PINNED_COMMIT}")
    normal_metadata = json.loads(
        (args.normal_root / "normal_cache_metadata.json").read_text(encoding="utf-8")
    )
    if normal_metadata.get("profile") != "offnet":
        raise ValueError("The authors' checkpoint requires the OFF-Net SNE profile")
    validate_formal_calibration_metadata(normal_metadata)
    dataset = ORFDNormalDataset(
        args.data_root, args.normal_root, "test", "offnet", args.height, args.width
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(args.device)
    bundle = load_offnet(args, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    bundle.model.load_state_dict(state, strict=True)
    metrics = evaluate("offnet", bundle.model, loader, device, amp=False)
    result = {
        "protocol": "authors' released OFF-Net checkpoint evaluated locally",
        "dataset": "ORFD",
        "partition": "testing",
        "samples": len(dataset),
        "input_size": [args.height, args.width],
        "metrics": metrics,
        "parameters": sum(parameter.numel() for parameter in bundle.model.parameters()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "strict_checkpoint_load": True,
        "official_source": str(args.official_source.resolve()),
        "official_commit": PINNED_COMMIT,
        "normal_cache_metadata": normal_metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
