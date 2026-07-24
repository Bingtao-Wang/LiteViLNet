#!/usr/bin/env python
"""Distill the LiteViLNet-Paper seed model into a lightweight LiteViLNet student."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from litevilnet.model_factory import MODEL_PRESETS, build_model
from litevilnet.models.losses import VLLiNetLoss
from litevilnet.data import AverageMeter, EarlyStopping, get_dataloader, save_checkpoint, set_seed
from litevilnet.metrics.deployment_metrics import BinarySegmentationMeter
from litevilnet.utils.common import system_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LiteViLNet edge distillation")
    parser.add_argument("--data_root", default="data/kitti_road")
    parser.add_argument("--train_split_file", default="")
    parser.add_argument("--val_split_file", default="")
    parser.add_argument("--teacher_preset", default="litevilnet_paper", choices=["litevilnet_paper"])
    parser.add_argument("--teacher_checkpoint", default=MODEL_PRESETS["litevilnet_paper"]["checkpoint_hint"])
    parser.add_argument("--student_checkpoint", default="", help="Optional student init checkpoint")
    parser.add_argument("--save_dir", default="weights/litevilnet/edge_distill")
    parser.add_argument("--log_dir", default="runs/train/edge_distill")
    parser.add_argument("--student_preset", default="litevilnet_edge", choices=["litevilnet_edge", "litevilnet_baseline"])
    parser.add_argument("--img_h", type=int, default=384)
    parser.add_argument("--img_w", type=int, default=1248)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulate_grad_batches", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seg_weight", type=float, default=1.0)
    parser.add_argument("--logit_weight", type=float, default=0.4)
    parser.add_argument("--boundary_weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument(
        "--skip_train_metrics",
        action="store_true",
        help="Skip the training-set threshold sweep; validation metrics are unchanged.",
    )
    parser.add_argument("--student_pretrained", action="store_true", default=True)
    parser.add_argument("--no_student_pretrained", action="store_false", dest="student_pretrained")
    return parser.parse_args()


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("distill")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(os.path.join(log_dir, "train.log"))):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def boundary_map(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    avg = F.avg_pool2d(probs, kernel_size=3, stride=1, padding=1)
    return torch.abs(probs - avg)


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    student = student_logits / temperature
    teacher = teacher_logits.detach() / temperature
    return F.binary_cross_entropy_with_logits(student, torch.sigmoid(teacher)) * (temperature**2)


def train_epoch(teacher, student, dataloader, criterion, optimizer, device, scaler, args):
    teacher.eval()
    student.train()
    loss_meter = AverageMeter()
    metrics = BinarySegmentationMeter()

    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(dataloader):
        rgb = batch["rgb"].to(device, non_blocking=True)
        adi = batch["adi"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                with torch.no_grad():
                    teacher_logits = teacher(rgb, adi)
                    if isinstance(teacher_logits, tuple):
                        teacher_logits = teacher_logits[0]
                student_out = student(rgb, adi, return_aux=True)
                if isinstance(student_out, tuple):
                    student_logits, aux = student_out
                else:
                    student_logits, aux = student_out, None
                seg_loss = criterion(student_logits, aux, label)
                logit_loss = distillation_loss(student_logits, teacher_logits, args.temperature)
                b_loss = F.l1_loss(boundary_map(student_logits), boundary_map(teacher_logits).detach())
                loss = args.seg_weight * seg_loss + args.logit_weight * logit_loss + args.boundary_weight * b_loss
            scaler.scale(loss / args.accumulate_grad_batches).backward()
            if (batch_index + 1) % args.accumulate_grad_batches == 0 or (batch_index + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                teacher_logits = teacher(rgb, adi)
                if isinstance(teacher_logits, tuple):
                    teacher_logits = teacher_logits[0]
            student_out = student(rgb, adi, return_aux=True)
            if isinstance(student_out, tuple):
                student_logits, aux = student_out
            else:
                student_logits, aux = student_out, None
            seg_loss = criterion(student_logits, aux, label)
            logit_loss = distillation_loss(student_logits, teacher_logits, args.temperature)
            b_loss = F.l1_loss(boundary_map(student_logits), boundary_map(teacher_logits).detach())
            loss = args.seg_weight * seg_loss + args.logit_weight * logit_loss + args.boundary_weight * b_loss
            (loss / args.accumulate_grad_batches).backward()
            if (batch_index + 1) % args.accumulate_grad_batches == 0 or (batch_index + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        loss_meter.update(loss.item(), rgb.size(0))
        if not args.skip_train_metrics:
            with torch.no_grad():
                metrics.update(student_logits, label, input_type="logits")

    return loss_meter.avg, None if args.skip_train_metrics else metrics.compute()


def validate(student, dataloader, device):
    student.eval()
    metrics = BinarySegmentationMeter()
    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            adi = batch["adi"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            logits = student(rgb, adi)
            if isinstance(logits, tuple):
                logits = logits[0]
            metrics.update(logits, label, input_type="logits")
    return metrics.compute()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic
    run_save_dir = os.path.join(args.save_dir, f"seed_{args.seed}")
    run_log_dir = os.path.join(args.log_dir, f"seed_{args.seed}")
    os.makedirs(run_save_dir, exist_ok=True)
    logger = setup_logging(run_log_dir)
    logger.info(f"Arguments: {args}")
    logger.info(f"Command: {' '.join(sys.argv)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher, teacher_meta = build_model(
        args.teacher_preset,
        checkpoint=args.teacher_checkpoint,
        device=device,
        strict=False,
        allow_partial=False,
    )
    student, student_meta = build_model(
        args.student_preset,
        checkpoint=args.student_checkpoint or None,
        device=device,
        pretrained=args.student_pretrained,
        deep_supervision=True,
        strict=False,
    )
    teacher.requires_grad_(False)
    logger.info(f"Teacher: {teacher_meta['label']} params={teacher_meta['parameters']/1e6:.2f}M")
    logger.info(f"Student: {student_meta['label']} params={student_meta['parameters']/1e6:.2f}M")

    train_loader = get_dataloader(
        args.data_root,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_h=args.img_h,
        img_w=args.img_w,
        use_augmentation=True,
        split_file=args.train_split_file or None,
        seed=args.seed,
        drop_last=args.drop_last,
    )
    val_loader = get_dataloader(
        args.data_root,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_h=args.img_h,
        img_w=args.img_w,
        use_augmentation=False,
        shuffle=False,
        split_file=args.val_split_file or None,
        seed=args.seed,
    )

    criterion = VLLiNetLoss(use_deep_supervision=True).to(device)
    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if args.amp and torch.cuda.is_available() else None
    early_stopping = EarlyStopping(patience=args.patience, mode="max")
    best = 0.0
    best_epoch = 0
    best_val_metrics = None

    for epoch in range(args.epochs):
        train_loss, train_metrics = train_epoch(teacher, student, train_loader, criterion, optimizer, device, scaler, args)
        val_metrics = validate(student, val_loader, device)
        scheduler.step()
        current = val_metrics["MaxF"]
        train_maxf = "skipped" if train_metrics is None else f"{train_metrics['MaxF']:.4f}"
        logger.info(
            "Epoch %03d/%03d | loss %.4f | train MaxF %s | val MaxF %.4f | AP %.4f | PRE %.4f | REC %.4f | FPR %.4f | FNR %.4f | BestThreshold %.2f",
            epoch + 1,
            args.epochs,
            train_loss,
            train_maxf,
            current,
            val_metrics["AP"],
            val_metrics["PRE"],
            val_metrics["REC"],
            val_metrics["FPR"],
            val_metrics["FNR"],
            val_metrics["BestThreshold"],
        )
        if current > best:
            best = current
            best_epoch = epoch + 1
            best_val_metrics = dict(val_metrics)
            save_checkpoint(student, optimizer, epoch + 1, best, os.path.join(run_save_dir, "best_model.pth"), args)
            logger.info("New best student saved: %.4f", best)
        save_checkpoint(student, optimizer, epoch + 1, best, os.path.join(run_save_dir, "latest_model.pth"), args)
        if early_stopping(current):
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    logger.info("Distillation complete. Best MaxF: %.4f", best)
    result = {
        "experiment": "logit-and-boundary knowledge distillation control",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "epochs_completed": epoch + 1,
        "best_val_metrics": best_val_metrics,
        "teacher": teacher_meta,
        "student": student_meta,
        "data_root": args.data_root,
        "train_split_file": args.train_split_file or None,
        "val_split_file": args.val_split_file or None,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "image_height": args.img_h,
        "image_width": args.img_w,
        "batch_size": args.batch_size,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "temperature": args.temperature,
        "seg_weight": args.seg_weight,
        "logit_weight": args.logit_weight,
        "boundary_weight": args.boundary_weight,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "amp": bool(args.amp and torch.cuda.is_available()),
        "deterministic": args.deterministic,
        "student_pretrained": args.student_pretrained,
        "skip_train_metrics": args.skip_train_metrics,
        "system": system_metadata(),
    }
    with open(os.path.join(run_save_dir, "result.json"), "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
