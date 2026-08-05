#!/usr/bin/env python3
"""Retrain pinned OFF-Net under the matched local KITTI protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from litevilnet.metrics.deployment_metrics import BinarySegmentationMeter
from tools.train_matched_kitti_baseline import seed_all, seed_worker
from tools.train_matched_orfd_baseline import load_offnet


PINNED_COMMIT = "50e63d24836198e8fb5af707e521f414104b4876"
EXPECTED_REMOTE = "https://github.com/chaytonmin/Off-Road-Freespace-Detection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normal-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--early-stop-validations", type=int, default=20)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--smoke-eval-samples", type=int, default=8)
    return parser.parse_args()


def git_value(path: Path, arguments: list[str]) -> str:
    return subprocess.check_output(["git", *arguments], cwd=path, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MatchedKITTIOFFNetDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        normal_root: Path,
        split: str,
        height: int,
        width: int,
    ) -> None:
        self.data_root = data_root
        self.normal_root = normal_root
        self.split = split
        self.height = height
        self.width = width
        self.images = sorted((data_root / split / "image_2").glob("*.png"))
        if not self.images:
            raise FileNotFoundError(data_root / split / "image_2")
        for image in self.images:
            stem = image.stem
            required = [self._label_path(stem), normal_root / split / "normal" / f"{stem}.npy"]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing OFF-Net KITTI inputs for {stem}: {missing}")

    def _label_path(self, stem: str) -> Path:
        category, frame = stem.split("_", 1)
        return self.data_root / self.split / "gt_image_2" / f"{category}_road_{frame}.png"

    @staticmethod
    def _label(path: Path) -> Image.Image:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        road = ((rgb[:, :, 2] > 200) & (rgb[:, :, 0] > 200)).astype(np.uint8)
        return Image.fromarray(road)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.images[index]
        stem = image_path.stem
        image = Image.open(image_path).convert("RGB")
        label = self._label(self._label_path(stem))
        normal = np.load(
            self.normal_root / self.split / "normal" / f"{stem}.npy", allow_pickle=False
        )
        if normal.shape[0] != 3 or normal.dtype != np.float32:
            raise ValueError(f"Invalid OFF-Net normal cache for {stem}")
        label_training = label.resize(
            (self.width // 4, self.height // 4), Image.Resampling.NEAREST
        )
        image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        label = label.resize((self.width, self.height), Image.Resampling.NEAREST)
        normal_hwc = cv2.resize(
            normal.transpose(1, 2, 0),
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb)),
            "normal": torch.from_numpy(
                np.ascontiguousarray(normal_hwc.astype(np.float32).transpose(2, 0, 1))
            ),
            "label": torch.from_numpy(np.asarray(label, dtype=np.int64).copy()),
            "label_training": torch.from_numpy(
                np.asarray(label_training, dtype=np.int64).copy()
            ),
            "name": stem,
        }

    def __len__(self) -> int:
        return len(self.images)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, limit: int = 0):
    model.eval()
    meter = BinarySegmentationMeter(thresholds=101)
    processed = 0
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        normal = batch["normal"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        probability = torch.softmax(model(rgb, normal), dim=1)[:, 1]
        meter.update(probability, label, input_type="prob")
        processed += rgb.shape[0]
        if limit and processed >= limit:
            break
    return {"samples": processed, "threshold_swept": meter.compute()}


def save_checkpoint(path: Path, bundle: Any, epoch: int, metric: float, args: argparse.Namespace):
    torch.save(
        {
            "model_state_dict": bundle.model.state_dict(),
            "optimizer_state_dict": bundle.optimizer.state_dict(),
            "scheduler_state_dict": bundle.scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": metric,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    source = args.official_source.resolve()
    commit = git_value(source, ["rev-parse", "HEAD"])
    remote = git_value(source, ["remote", "get-url", "origin"])
    if commit != PINNED_COMMIT or remote != EXPECTED_REMOTE:
        raise RuntimeError(f"Unexpected OFF-Net source: {remote}@{commit}")
    if git_value(source, ["status", "--porcelain"]):
        raise RuntimeError("OFF-Net official source is not clean")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda")
    normal_metadata = json.loads(
        (args.normal_root / "normal_cache_metadata.json").read_text(encoding="utf-8")
    )
    if normal_metadata.get("profile") != "offnet":
        raise ValueError("KITTI OFF-Net requires normals generated by the OFF-Net SNE profile")

    datasets = {
        split: MatchedKITTIOFFNetDataset(
            args.data_root, args.normal_root, split, args.height, args.width
        )
        for split in ("training", "validation")
    }
    train_loader = DataLoader(
        datasets["training"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        datasets["validation"],
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed + 1),
        persistent_workers=args.num_workers > 0,
    )
    bundle = load_offnet(args, device)
    torch.cuda.reset_peak_memory_stats(device)
    start_epoch = 0
    best_metric = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        bundle.model.load_state_dict(checkpoint["model_state_dict"])
        bundle.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        bundle.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_metric = float(checkpoint["best_metric"])

    train_start = time.time()
    global_step = 0
    stale = 0
    log_path = args.output_dir / "train_metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        bundle.model.train()
        losses: list[float] = []
        epoch_start = time.time()
        learning_rate = float(bundle.optimizer.param_groups[0]["lr"])
        for batch in train_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            normal = batch["normal"].to(device, non_blocking=True)
            target = batch["label_training"].to(device, non_blocking=True)
            bundle.optimizer.zero_grad(set_to_none=True)
            logits = bundle.model(rgb, normal)
            if target.shape[-2:] != logits.shape[-2:]:
                raise ValueError(
                    f"Training target/output mismatch: {target.shape[-2:]} vs {logits.shape[-2:]}"
                )
            loss = F.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite OFF-Net loss at epoch {epoch + 1}")
            loss.backward()
            bundle.optimizer.step()
            losses.append(float(loss.detach().cpu()))
            global_step += 1
            if args.smoke_steps and global_step >= args.smoke_steps:
                break
        bundle.scheduler.step()
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "train_loss": float(np.mean(losses)),
            "epoch_seconds": time.time() - epoch_start,
            "global_step": global_step,
        }
        should_validate = (
            args.smoke_steps > 0
            or (epoch + 1) % args.val_every == 0
            or epoch + 1 == args.epochs
        )
        if should_validate:
            limit = args.smoke_eval_samples if args.smoke_steps else 0
            validation = evaluate(bundle.model, val_loader, device, limit)
            record["validation"] = validation
            current = float(validation["threshold_swept"]["MaxF"])
            if current > best_metric:
                best_metric = current
                stale = 0
                save_checkpoint(
                    args.output_dir / "best_model.pth", bundle, epoch + 1, best_metric, args
                )
            else:
                stale += 1
            record["best_MaxF"] = best_metric
            record["stale_validations"] = stale
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.smoke_steps and global_step >= args.smoke_steps:
            break
        if args.early_stop_validations and stale >= args.early_stop_validations:
            break

    best_path = args.output_dir / "best_model.pth"
    checkpoint = torch.load(best_path, map_location=device)
    bundle.model.load_state_dict(checkpoint["model_state_dict"])
    limit = args.smoke_eval_samples if args.smoke_steps else 0
    validation = evaluate(bundle.model, val_loader, device, limit)
    metrics = validation["threshold_swept"]
    result = {
        "protocol": "matched local KITTI perspective-view retraining",
        "baseline": "offnet",
        "seed": args.seed,
        "input_size": [args.height, args.width],
        "train_count": len(datasets["training"]),
        "val_count": len(datasets["validation"]),
        "epochs_requested": args.epochs,
        "best_epoch": int(checkpoint["epoch"]),
        "batch_size": args.batch_size,
        "amp": False,
        "metrics": metrics,
        "validation": validation,
        "parameters": sum(parameter.numel() for parameter in bundle.model.parameters()),
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256(best_path),
        "official_source": str(source),
        "official_remote": remote,
        "official_commit": commit,
        "official_semantic_diff_clean": True,
        "official_file_sha256": {
            str(path.resolve().relative_to(source)): sha256(path) for path in bundle.official_files
        },
        "normal_cache_metadata": normal_metadata,
        "split_metadata": json.loads(
            (args.data_root / "matched_split_metadata.json").read_text(encoding="utf-8")
        ),
        "runtime": {
            "seconds": time.time() - train_start,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
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
