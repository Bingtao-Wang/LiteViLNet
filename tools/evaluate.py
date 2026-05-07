#!/usr/bin/env python
"""Evaluate VLLiNet presets with true threshold-swept MaxF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from litevilnet.utils.common import append_csv, system_metadata, write_json
from litevilnet.metrics.deployment_metrics import BinarySegmentationMeter
from litevilnet.model_factory import MODEL_PRESETS, available_presets, build_model
from litevilnet.data import get_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LiteViLNet presets")
    parser.add_argument("--preset", default="vllinet_paper", choices=available_presets())
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data_root", default="data/kitti_road")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--category", default="all")
    parser.add_argument("--img_h", type=int, default=384)
    parser.add_argument("--img_w", type=int, default=1248)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--thresholds", type=int, default=101)
    parser.add_argument("--output", default="runs/eval/eval.json")
    parser.add_argument("--csv", default="runs/eval/eval_summary.csv")
    parser.add_argument("--allow_partial", action="store_true", help="Allow non-aux partial checkpoint loading")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or MODEL_PRESETS[args.preset].get("checkpoint_hint", "")
    checkpoint = checkpoint if checkpoint and Path(checkpoint).exists() else None
    model, metadata = build_model(
        args.preset,
        checkpoint=checkpoint,
        device=device,
        strict=False,
        allow_partial=args.allow_partial,
    )

    if args.precision == "fp16":
        model.half()

    dataloader = get_dataloader(
        data_root=args.data_root,
        split=args.split,
        category=args.category,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_h=args.img_h,
        img_w=args.img_w,
        use_augmentation=False,
        shuffle=False,
    )

    meter = BinarySegmentationMeter(thresholds=args.thresholds)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {metadata['label']}"):
            rgb = batch["rgb"].to(device, non_blocking=True)
            adi = batch["adi"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            if args.precision == "fp16":
                rgb = rgb.half()
                adi = adi.half()
            logits = model(rgb, adi)
            if isinstance(logits, tuple):
                logits = logits[0]
            meter.update(logits, label, input_type="logits")

    metrics = meter.compute()
    result = {
        "backend": "pytorch",
        "preset": args.preset,
        "label": metadata["label"],
        "checkpoint": checkpoint,
        "precision": args.precision,
        "data_root": args.data_root,
        "split": args.split,
        "input_shape": [args.batch_size, 3, args.img_h, args.img_w],
        "parameters": metadata["parameters"],
        "metrics": metrics,
        "system": system_metadata(),
    }
    write_json(args.output, result)
    append_csv(
        args.csv,
        {
            "backend": "pytorch",
            "model": metadata["label"],
            "preset": args.preset,
            "precision": args.precision,
            "split": args.split,
            "MaxF": metrics["MaxF"],
            "BestThreshold": metrics["BestThreshold"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "IoU": metrics["IoU"],
            "parameters_M": metadata["parameters"] / 1e6,
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
