#!/usr/bin/env python
"""Train LiteViLNet RGB+Depth on manually labeled robot walkable-path frames."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from litevilnet.data import AverageMeter, save_checkpoint, set_seed
from litevilnet.data.robot_road_dataset import RobotRoadRGBDepthDataset
from litevilnet.metrics.deployment_metrics import BinarySegmentationMeter
from litevilnet.models.litevilnet_rgbdepth import LiteViLNetRGBDepth
from litevilnet.models.losses import VLLiNetLoss
from litevilnet.utils.common import system_metadata, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LiteViLNet RGB+Depth robot path smoke training")
    parser.add_argument("--data_root", default="data/robot_road")
    parser.add_argument("--session", default="session_20260509_172746")
    parser.add_argument("--split_file", default="data/robot_road/splits/seed18_labeled.txt")
    parser.add_argument("--mask_dir", default="", help="Optional directory with binary manual masks")
    parser.add_argument("--img_h", type=int, default=384)
    parser.add_argument("--img_w", type=int, default=608)
    parser.add_argument("--max_depth_mm", type=float, default=12000.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--use_deep_supervision", action="store_true", default=True)
    parser.add_argument("--no_deep_supervision", action="store_false", dest="use_deep_supervision")
    parser.add_argument("--save_dir", default="output/3.0_litevilnet_rgbdepth_robot_path_seed18_full_smoke")
    parser.add_argument("--log_dir", default="output/3.0_litevilnet_rgbdepth_robot_path_seed18_full_smoke")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("train_robot_rgbdepth")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(Path(log_dir) / "train.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_readme(exp_dir: Path, args: argparse.Namespace) -> None:
    experiment_name = exp_dir.name
    readme = f"""# {experiment_name}

## Purpose

Train a separate LiteViLNet RGB+Depth branch on manually labeled robot
walkable-path frames listed in the split file. This is a smoke/pseudo-label
bootstrapping run, not a generalization benchmark.

## Setup

- Dataset: `{args.data_root}`
- Session: `{args.session}`
- Split file: `{args.split_file}`
- Mask dir: `{args.mask_dir or 'default: <data_root>/annotations/manual_masks'}`
- Input: RGB + encoded aligned depth3
- Frames: all labeled frames in split file
- Validation: none
- Selection metric: train MaxF
- Image size: {args.img_h}x{args.img_w}
- Epochs: {args.epochs}
- Batch size: {args.batch_size}
- LR: {args.lr}
- Weight decay: {args.weight_decay}
- AMP: {bool(args.amp)}

## Outputs

- `best_model.pth`
- `latest_model.pth`
- `train.log`
- `train_metrics.json`
- `config.json`
"""
    (exp_dir / "README.md").write_text(readme, encoding="utf-8")


def train_epoch(model, dataloader, criterion, optimizer, device, scaler) -> tuple[float, dict[str, float]]:
    model.train()
    loss_meter = AverageMeter()
    metrics = BinarySegmentationMeter()

    for batch in dataloader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth3 = batch["depth3"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(rgb, depth3, return_aux=True)
                if isinstance(output, tuple):
                    logits, aux_outputs = output
                    loss = criterion(logits, aux_outputs, label)
                else:
                    logits = output
                    loss = criterion(logits, None, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(rgb, depth3, return_aux=True)
            if isinstance(output, tuple):
                logits, aux_outputs = output
                loss = criterion(logits, aux_outputs, label)
            else:
                logits = output
                loss = criterion(logits, None, label)
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), rgb.size(0))
        with torch.no_grad():
            if logits.shape[-2:] != label.shape[-2:]:
                probs = torch.sigmoid(F.interpolate(logits, size=label.shape[-2:], mode="bilinear", align_corners=False))
            else:
                probs = torch.sigmoid(logits)
            metrics.update(probs, label, input_type="prob")

    return loss_meter.avg, metrics.compute()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    exp_dir = Path(args.save_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.log_dir)
    write_json(exp_dir / "config.json", vars(args))
    write_readme(exp_dir, args)

    logger.info("Arguments: %s", args)
    logger.info("Command: %s", " ".join(sys.argv))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    logger.info("Device: %s", device)

    dataset = RobotRoadRGBDepthDataset(
        data_root=args.data_root,
        session=args.session,
        split_file=args.split_file,
        img_h=args.img_h,
        img_w=args.img_w,
        max_depth_mm=args.max_depth_mm,
        require_mask=True,
        mask_dir=args.mask_dir or None,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    logger.info("Dataset frames: %d", len(dataset))

    model = LiteViLNetRGBDepth(
        pretrained=args.pretrained,
        use_deep_supervision=args.use_deep_supervision,
    ).to(device)
    criterion = VLLiNetLoss(use_deep_supervision=args.use_deep_supervision).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None

    best_metric = -1.0
    history = []
    for epoch in range(args.epochs):
        logger.info("\n%s Epoch %d/%d %s", "=" * 32, epoch + 1, args.epochs, "=" * 32)
        train_loss, train_metrics = train_epoch(model, dataloader, criterion, optimizer, device, scaler)
        scheduler.step()

        row = {"epoch": epoch + 1, "train_loss": train_loss, **train_metrics}
        history.append(row)
        logger.info(
            "Train Loss : %.4f | MaxF %.4f | AP %.4f | PRE %.4f | REC %.4f | FPR %.4f | FNR %.4f | BestThreshold %.2f",
            train_loss,
            train_metrics["MaxF"],
            train_metrics["AP"],
            train_metrics["PRE"],
            train_metrics["REC"],
            train_metrics["FPR"],
            train_metrics["FNR"],
            train_metrics["BestThreshold"],
        )

        if train_metrics["MaxF"] > best_metric:
            best_metric = train_metrics["MaxF"]
            save_checkpoint(model, optimizer, epoch + 1, best_metric, exp_dir / "best_model.pth", args)
            logger.info("New best model saved. Train MaxF: %.4f", best_metric)
        save_checkpoint(model, optimizer, epoch + 1, best_metric, exp_dir / "latest_model.pth", args)

        write_json(
            exp_dir / "train_metrics.json",
            {
                "best_metric": best_metric,
                "best_epoch": max(history, key=lambda x: x["MaxF"])["epoch"],
                "latest": row,
                "history": history,
                "system": system_metadata(),
            },
        )

    logger.info("Training completed. Best train MaxF: %.4f", best_metric)


if __name__ == "__main__":
    main()
