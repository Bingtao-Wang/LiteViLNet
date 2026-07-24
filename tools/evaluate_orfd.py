#!/usr/bin/env python
"""Evaluate an ablation checkpoint on a released ORFD partition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from litevilnet.data import get_orfd_dataloader
from litevilnet.metrics.deployment_metrics import BinarySegmentationMeter
from litevilnet.models.vllinet_ablation import get_ablation_model
from litevilnet.utils.common import system_metadata, write_json


CONFIGS = (
    "baseline",
    "add_lidar",
    "add_fusion",
    "add_bridge",
    "add_deep_sup",
    "full",
    "optimal",
    "transformer_bridge",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=CONFIGS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--img-h", type=int, default=704)
    parser.add_argument("--img-w", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", default="fp16", choices=("fp32", "fp16"))
    parser.add_argument("--thresholds", type=int, default=101)
    parser.add_argument("--seed", type=int, default=42, help="DataLoader seed; evaluation order only")
    parser.add_argument("--quiet", action="store_true", help="Disable the progress bar")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_orfd_metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "GlobalAcc": (tp + tn) / (tp + fp + tn + fn + eps),
        "PRE": precision,
        "REC": recall,
        "F_score": 2.0 * precision * recall / (precision + recall + eps),
        "IoU": tp / (tp + fp + fn + eps),
        "threshold": 0.5,
        **counts,
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = get_ablation_model(args.config, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    if args.precision == "fp16":
        model.half()

    loader = get_orfd_dataloader(
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_h=args.img_h,
        img_w=args.img_w,
        use_augmentation=False,
        seed=args.seed,
        drop_last=False,
        shuffle=False,
        return_original_label=True,
    )
    meter = BinarySegmentationMeter(thresholds=args.thresholds)
    official_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"ORFD {args.split}: {args.config}", disable=args.quiet):
            rgb = batch["rgb"].to(device, non_blocking=True)
            geometry = batch["adi"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            if args.precision == "fp16":
                rgb = rgb.half()
                geometry = geometry.half()
            logits = model(rgb, geometry)
            if isinstance(logits, tuple):
                logits = logits[0]
            meter.update(logits, label, input_type="logits")
            # Match the released OFF-Net test protocol: turn model output into
            # a discrete binary prediction, resize it with nearest-neighbor to
            # the original 1280x720 label, and accumulate one confusion matrix
            # over the complete testing partition.  For a one-logit BCE model,
            # argmax over [background, foreground] is equivalent to logit > 0.
            logits_at_input = F.interpolate(
                logits.float(),
                size=(args.img_h, args.img_w),
                mode="bilinear",
                align_corners=False,
            )
            prediction = logits_at_input > 0.0
            label_original = batch["label_original"].to(device, non_blocking=True).bool()
            prediction = F.interpolate(
                prediction.float(),
                size=label_original.shape[-2:],
                mode="nearest",
            ).squeeze(1).bool()
            official_counts["tp"] += int((prediction & label_original).sum().item())
            official_counts["fp"] += int((prediction & ~label_original).sum().item())
            official_counts["tn"] += int((~prediction & ~label_original).sum().item())
            official_counts["fn"] += int((~prediction & label_original).sum().item())

    result = {
        "dataset": "ORFD",
        "partition": args.split,
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_best_metric": checkpoint.get("best_metric") if isinstance(checkpoint, dict) else None,
        "strict_checkpoint_load": True,
        "samples": len(loader.dataset),
        "input_shape_per_modality": [args.batch_size, 3, args.img_h, args.img_w],
        "precision": args.precision,
        "thresholds": args.thresholds,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "metrics": meter.compute(),
        "official_metrics": official_orfd_metrics(official_counts),
        "system": system_metadata(),
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
