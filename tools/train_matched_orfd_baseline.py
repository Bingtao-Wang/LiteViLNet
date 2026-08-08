#!/usr/bin/env python3
"""Retrain official USNet, SNE-RoadSeg, or OFF-Net on released ORFD splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from litevilnet.metrics.deployment_metrics import BinarySegmentationMeter
from tools.cache_official_orfd_normals import validate_formal_calibration_metadata
from tools.train_matched_kitti_baseline import (
    ModelBundle,
    load_sne_roadseg,
    load_usnet,
    seed_all,
    seed_worker,
    update_usnet_lr,
    usnet_loss,
)


RGB_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(3, 1, 1)
RGB_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(3, 1, 1)
PINNED_COMMITS = {
    "usnet": "d761158ad42df7dcb62fa257dd02ce11c85f94a5",
    "sne_roadseg": "5e7900bfd59887634ced687ffe85a73018a38659",
    "offnet": "50e63d24836198e8fb5af707e521f414104b4876",
}
EXPECTED_REMOTES = {
    "usnet": "https://github.com/morancyc/USNet.git",
    "sne_roadseg": "https://github.com/hlwang1124/SNE-RoadSeg.git",
    "offnet": "https://github.com/chaytonmin/Off-Road-Freespace-Detection",
}
SPLITS = {"train": "training", "val": "validation", "test": "testing"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=tuple(PINNED_COMMITS), required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normal-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help=(
            "Physical batches per optimizer update. By default OFF-Net uses 4 "
            "to reproduce its released global batch 8 from physical batch 2; "
            "the other baselines use 1."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--early-stop-validations", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--smoke-eval-samples", type=int, default=8)
    return parser.parse_args()


def git_value(path: Path, arguments: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ORFDNormalDataset(Dataset):
    """Official ORFD partition adapter with method-specific released transforms."""

    def __init__(
        self,
        data_root: Path,
        normal_root: Path,
        split: str,
        baseline: str,
        height: int,
        width: int,
    ) -> None:
        self.data_root = data_root
        self.normal_root = normal_root
        self.split = split
        self.baseline = baseline
        self.height = height
        self.width = width
        self.training = split == "train"
        released_split = SPLITS[split]
        self.images = sorted((data_root / released_split / "image_data").glob("*.png"))
        if not self.images:
            raise FileNotFoundError(data_root / released_split / "image_data")
        for image_path in self.images:
            stem = image_path.stem
            required = [
                data_root / released_split / "gt_image" / f"{stem}_fillcolor.png",
                normal_root / released_split / "normal" / f"{stem}.npy",
            ]
            if self.training and baseline == "usnet":
                required.append(normal_root / released_split / "normal_flipped" / f"{stem}.npy")
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing ORFD inputs for {stem}: {missing}")

    @staticmethod
    def _label(path: Path) -> Image.Image:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        return Image.fromarray((rgb[:, :, 2] > 200).astype(np.uint8))

    @staticmethod
    def _gaussian_noise(image: Image.Image, sigma: float = 10.0) -> Image.Image:
        array = np.asarray(image).astype(np.int32)
        for channel in range(3):
            noise = np.random.normal(0.0, sigma, array[:, :, channel].size)
            values = np.clip(array[:, :, channel].reshape(-1) + noise, 0, 255)
            array[:, :, channel] = values.reshape(array.shape[:2])
        return Image.fromarray(array.astype(np.uint8))

    @staticmethod
    def _color_jitter(image: Image.Image) -> Image.Image:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.6, 1.4))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.6, 1.4))
        return ImageEnhance.Color(image).enhance(random.uniform(0.6, 1.4))

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.images[index]
        stem = image_path.stem
        released_split = SPLITS[self.split]
        image = Image.open(image_path).convert("RGB")
        label_original = self._label(
            self.data_root / released_split / "gt_image" / f"{stem}_fillcolor.png"
        )
        label = label_original

        flipped = self.training and self.baseline == "usnet" and random.random() < 0.5
        normal_dir = "normal_flipped" if flipped else "normal"
        normal = np.load(
            self.normal_root / released_split / normal_dir / f"{stem}.npy",
            allow_pickle=False,
        )
        if normal.shape[0] != 3 or normal.dtype != np.float32:
            raise ValueError(f"Invalid normal cache for {stem}: {normal.shape}, {normal.dtype}")
        if flipped:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if self.training and self.baseline == "usnet":
            if random.random() < 0.5:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))
            if random.random() < 0.5:
                image = self._gaussian_noise(image)

        label_training = (
            label.resize(
                (self.width // 4, self.height // 4), Image.Resampling.NEAREST
            )
            if self.baseline == "offnet"
            else label.resize((self.width, self.height), Image.Resampling.NEAREST)
        )
        image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        label = label.resize((self.width, self.height), Image.Resampling.NEAREST)
        normal_hwc = np.transpose(normal, (1, 2, 0))
        normal_hwc = cv2.resize(
            normal_hwc, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        if self.training and self.baseline == "usnet":
            image = self._color_jitter(image)

        rgb = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        if self.baseline == "usnet":
            rgb = (rgb - RGB_MEAN) / RGB_STD
        geometry = np.asarray(normal_hwc, dtype=np.float32).transpose(2, 0, 1)
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb)),
            "normal": torch.from_numpy(np.ascontiguousarray(geometry)),
            "label": torch.from_numpy(np.asarray(label, dtype=np.int64).copy()),
            "label_training": torch.from_numpy(
                np.asarray(label_training, dtype=np.int64).copy()
            ),
            "label_original": torch.from_numpy(
                np.asarray(label_original, dtype=np.int64).copy()
            ),
            "name": stem,
        }

    def __len__(self) -> int:
        return len(self.images)


def load_offnet(args: argparse.Namespace, device: torch.device) -> ModelBundle:
    source = args.official_source.resolve()
    sys.path.insert(0, str(source))
    from models.transformer_models.backbones import transformer  # pylint: disable=import-outside-toplevel

    model = transformer.define_RoadSeg(
        2, use_sne=True, init_type="kaiming", init_gain=0.02, gpu_ids=[]
    ).to(device)
    base_lr = args.learning_rate if args.learning_rate is not None else 1e-3
    optimizer = torch.optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 0.9 ** ((epoch + 1) // 25)
    )
    return ModelBundle(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        base_lr=base_lr,
        official_files=[
            source / "models" / "transformer_models" / "backbones" / "transformer.py",
            source / "models" / "transformer_models" / "decode_heads" / "head.py",
            source / "models" / "roadseg_model.py",
            source / "models" / "loss.py",
            source / "models" / "sne_model.py",
            source / "data" / "ORFD_dataset.py",
            source / "options" / "base_options.py",
            source / "options" / "train_options.py",
            source / "scripts" / "train.sh",
            source / "train.py",
        ],
    )


def forward_probability(
    baseline: str, model: torch.nn.Module, rgb: torch.Tensor, normal: torch.Tensor
) -> tuple[torch.Tensor, Any]:
    output = model(rgb, normal)
    if baseline == "usnet":
        alpha = output[-1]
        probability = alpha / alpha.sum(dim=1, keepdim=True)
        return probability[:, 1].reshape(
            rgb.shape[0], rgb.shape[2], rgb.shape[3]
        ), output
    return torch.softmax(output, dim=1)[:, 1], output


def fixed_metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "F_score": 2.0 * precision * recall / (precision + recall + eps),
        "PRE": precision,
        "REC": recall,
        "IoU": tp / (tp + fp + fn + eps),
        "GlobalAcc": (tp + tn) / (tp + fp + tn + fn + eps),
        "threshold": 0.5,
        **counts,
    }


@torch.no_grad()
def evaluate(
    baseline: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    max_samples: int = 0,
) -> dict[str, Any]:
    model.eval()
    meter = BinarySegmentationMeter(thresholds=101)
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    authors_counts = (
        {"tp": 0, "fp": 0, "tn": 0, "fn": 0} if baseline == "offnet" else None
    )
    processed = 0
    fixed_argmax_ground_truth_grid: list[int] | None = None
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        normal = batch["normal"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            probability, _ = forward_probability(baseline, model, rgb, normal)
        meter.update(probability, label, input_type="prob")
        prediction_native = probability > 0.5
        label_original = batch["label_original"].to(device, non_blocking=True).bool()
        fixed_argmax_ground_truth_grid = [
            int(value) for value in label_original.shape[-2:]
        ]
        prediction = F.interpolate(
            prediction_native.float().unsqueeze(1),
            size=label_original.shape[-2:],
            mode="nearest",
        ).squeeze(1).bool()
        counts["tp"] += int((prediction & label_original).sum().item())
        counts["fp"] += int((prediction & ~label_original).sum().item())
        counts["tn"] += int((~prediction & ~label_original).sum().item())
        counts["fn"] += int((~prediction & label_original).sum().item())
        if authors_counts is not None:
            # OFF-Net's released test.py restores both its native 1/4-scale
            # prediction and its 1/4-scale loader target to the original size.
            label_native = batch["label_training"].to(device, non_blocking=True).bool()
            if prediction_native.shape != label_native.shape:
                raise ValueError(
                    "OFF-Net native prediction/target mismatch: "
                    f"{prediction_native.shape} vs {label_native.shape}"
                )
            prediction_authors = F.interpolate(
                prediction_native.float().unsqueeze(1),
                size=label_original.shape[-2:],
                mode="nearest",
            ).squeeze(1).bool()
            label_authors = F.interpolate(
                label_native.float().unsqueeze(1),
                size=label_original.shape[-2:],
                mode="nearest",
            ).squeeze(1).bool()
            authors_counts["tp"] += int((prediction_authors & label_authors).sum().item())
            authors_counts["fp"] += int((prediction_authors & ~label_authors).sum().item())
            authors_counts["tn"] += int((~prediction_authors & ~label_authors).sum().item())
            authors_counts["fn"] += int((~prediction_authors & label_authors).sum().item())
        processed += rgb.shape[0]
        if max_samples and processed >= max_samples:
            break
    result = {
        "samples": min(processed, max_samples) if max_samples else processed,
        "threshold_swept_grid": [
            int(loader.dataset.height),
            int(loader.dataset.width),
        ],
        "fixed_argmax_ground_truth_grid": fixed_argmax_ground_truth_grid,
        "threshold_swept": meter.compute(),
        "official_fixed_argmax": fixed_metrics(counts),
    }
    if authors_counts is not None:
        result["authors_released_fixed_argmax"] = fixed_metrics(authors_counts)
    return result


def save_checkpoint(
    path: Path,
    bundle: ModelBundle,
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model_state_dict": bundle.model.state_dict(),
            "optimizer_state_dict": bundle.optimizer.state_dict(),
            "scheduler_state_dict": (
                bundle.scheduler.state_dict() if bundle.scheduler is not None else None
            ),
            "epoch": epoch,
            "best_metric": best_metric,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.amp and not args.device.startswith("cuda"):
        raise ValueError("--amp requires CUDA")
    gradient_accumulation_steps = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else (4 if args.baseline == "offnet" else 1)
    )
    if gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be positive")
    args.gradient_accumulation_steps = gradient_accumulation_steps
    commit = git_value(args.official_source, ["rev-parse", "HEAD"])
    remote = git_value(args.official_source, ["remote", "get-url", "origin"])
    status = git_value(args.official_source, ["status", "--porcelain"])
    if commit != PINNED_COMMITS[args.baseline]:
        raise RuntimeError(f"Expected {PINNED_COMMITS[args.baseline]}, found {commit}")
    if remote != EXPECTED_REMOTES[args.baseline]:
        raise RuntimeError(f"Unexpected official remote: {remote}")
    if status:
        raise RuntimeError(f"Official source is not clean: {args.official_source}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device(args.device)
    normal_metadata = json.loads(
        (args.normal_root / "normal_cache_metadata.json").read_text(encoding="utf-8")
    )
    expected_profile = "offnet" if args.baseline == "offnet" else "sne_roadseg"
    if normal_metadata.get("profile") != expected_profile:
        raise ValueError(
            f"{args.baseline} requires {expected_profile} normals, found "
            f"{normal_metadata.get('profile')!r}"
        )
    validate_formal_calibration_metadata(normal_metadata)

    datasets = {
        split: ORFDNormalDataset(
            args.data_root,
            args.normal_root,
            split,
            args.baseline,
            args.height,
            args.width,
        )
        for split in ("train", "val", "test")
    }
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
        persistent_workers=args.num_workers > 0,
    )
    eval_loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(args.seed + offset),
            persistent_workers=args.num_workers > 0,
        )
        for split, offset in (("val", 1), ("test", 2))
    }

    if args.baseline == "usnet":
        bundle = load_usnet(args, device)
    elif args.baseline == "sne_roadseg":
        bundle = load_sne_roadseg(args, device)
    else:
        bundle = load_offnet(args, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    start_epoch = 0
    best_metric = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        bundle.model.load_state_dict(checkpoint["model_state_dict"])
        bundle.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if bundle.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            bundle.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_metric = float(checkpoint["best_metric"])

    train_start = time.time()
    # Checkpoints written before resume-counter metadata was introduced still
    # resume exactly at an epoch boundary.  Reconstruct the completed physical
    # and optimizer steps from the fixed loader length and accumulation rule so
    # strict formal summaries retain the requested 30-epoch accounting.
    global_step = start_epoch * len(train_loader)
    optimizer_steps = start_epoch * (
        (len(train_loader) + gradient_accumulation_steps - 1)
        // gradient_accumulation_steps
    )
    stale_validations = 0
    last_validation: dict[str, Any] | None = None
    log_path = args.output_dir / "train_metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        bundle.model.train()
        if args.baseline == "usnet":
            learning_rate = update_usnet_lr(bundle.optimizer, bundle.base_lr, epoch, args.epochs)
        else:
            learning_rate = float(bundle.optimizer.param_groups[0]["lr"])
        losses: list[float] = []
        epoch_start = time.time()
        bundle.optimizer.zero_grad(set_to_none=True)
        planned_batches = (
            min(len(train_loader), args.smoke_steps)
            if args.smoke_steps
            else len(train_loader)
        )
        for batch_index, batch in enumerate(train_loader):
            rgb = batch["rgb"].to(device, non_blocking=True)
            normal = batch["normal"].to(device, non_blocking=True)
            label = batch["label_training"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                _, output = forward_probability(args.baseline, bundle.model, rgb, normal)
                if args.baseline == "usnet":
                    loss = usnet_loss(output, label, epoch, args.official_source)
                else:
                    if output.shape[-2:] != label.shape[-2:]:
                        raise ValueError(
                            f"Training target/output mismatch: {label.shape[-2:]} vs {output.shape[-2:]}"
                        )
                    loss = F.cross_entropy(output, label)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1}, batch {batch_index + 1}"
                )
            group_start = (
                batch_index // gradient_accumulation_steps
            ) * gradient_accumulation_steps
            group_size = min(
                gradient_accumulation_steps, planned_batches - group_start
            )
            scaler.scale(loss / group_size).backward()
            should_step = (
                (batch_index + 1) % gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
                or (args.smoke_steps and global_step + 1 >= args.smoke_steps)
            )
            if should_step:
                scaler.step(bundle.optimizer)
                scaler.update()
                bundle.optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            losses.append(float(loss.detach().cpu()))
            global_step += 1
            if args.smoke_steps and global_step >= args.smoke_steps:
                break
        if bundle.scheduler is not None:
            bundle.scheduler.step()

        should_validate = (
            args.smoke_steps > 0
            or (epoch + 1) % args.val_every == 0
            or epoch + 1 == args.epochs
        )
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "train_loss": float(np.mean(losses)),
            "epoch_seconds": time.time() - epoch_start,
            "global_step": global_step,
            "optimizer_steps": optimizer_steps,
        }
        if should_validate:
            limit = args.smoke_eval_samples if args.smoke_steps else 0
            last_validation = evaluate(
                args.baseline,
                bundle.model,
                eval_loaders["val"],
                device,
                args.amp,
                max_samples=limit,
            )
            record["validation"] = last_validation
            current = float(last_validation["threshold_swept"]["MaxF"])
            if current > best_metric:
                best_metric = current
                stale_validations = 0
                save_checkpoint(args.output_dir / "best_model.pth", bundle, epoch + 1, best_metric, args)
            else:
                stale_validations += 1
            record["best_validation_F_score"] = best_metric
            record["stale_validations"] = stale_validations
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.smoke_steps and global_step >= args.smoke_steps:
            break
        if args.early_stop_validations and stale_validations >= args.early_stop_validations:
            break

    best_path = args.output_dir / "best_model.pth"
    checkpoint = torch.load(best_path, map_location=device)
    bundle.model.load_state_dict(checkpoint["model_state_dict"])
    eval_limit = args.smoke_eval_samples if args.smoke_steps else 0
    validation = evaluate(
        args.baseline, bundle.model, eval_loaders["val"], device, args.amp, eval_limit
    )
    testing = evaluate(
        args.baseline, bundle.model, eval_loaders["test"], device, args.amp, eval_limit
    )
    result = {
        "protocol": "local ORFD released-split retraining",
        "baseline": args.baseline,
        "seed": args.seed,
        "input_size": [args.height, args.width],
        "train_count": len(datasets["train"]),
        "val_count": len(datasets["val"]),
        "test_count": len(datasets["test"]),
        "epochs_requested": args.epochs,
        "best_epoch": int(checkpoint["epoch"]),
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * gradient_accumulation_steps,
        "physical_training_steps": global_step,
        "optimizer_steps": optimizer_steps,
        "amp": args.amp,
        "checkpoint_selection": "validation 101-threshold MaxF",
        "validation": validation,
        "testing": testing,
        "parameters": sum(parameter.numel() for parameter in bundle.model.parameters()),
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256(best_path),
        "official_source": str(args.official_source.resolve()),
        "official_remote": remote,
        "official_commit": commit,
        "official_semantic_diff_clean": True,
        "official_file_sha256": {
            str(path.resolve().relative_to(args.official_source.resolve())): sha256(path)
            for path in bundle.official_files
        },
        "normal_cache_metadata": normal_metadata,
        "runtime": {
            "seconds": time.time() - train_start,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "command": sys.argv,
            "peak_allocated_cuda_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        },
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
