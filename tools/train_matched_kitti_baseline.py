#!/usr/bin/env python3
"""Retrain official USNet, SNE-RoadSeg, or PLARD under matched KITTI protocol.

The model definitions, losses, ImageNet initialization, and optimizer recipes
come from the cited official repositories.  This adapter changes only the
dataset split plumbing, caches deterministic official SNE normals, and applies
the same 101-threshold perspective-view evaluator used for LiteViLNet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
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


RGB_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(3, 1, 1)
RGB_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(3, 1, 1)
PINNED_COMMITS = {
    "usnet": "d761158ad42df7dcb62fa257dd02ce11c85f94a5",
    "sne_roadseg": "5e7900bfd59887634ced687ffe85a73018a38659",
    "plard": "44485803092e729661c696ab6c03f6f2fabc8701",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", choices=("usnet", "sne_roadseg", "plard"), required=True
    )
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--early-stop-validations", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--allow-modified-official-source", action="store_true")
    parser.add_argument("--allow-unpinned-official-source", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(path: Path, arguments: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def official_source_is_semantically_clean(path: Path) -> bool:
    completed = subprocess.run(
        ["git", "diff", "--ignore-space-at-eol", "--exit-code", "HEAD", "--"],
        cwd=path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


class MatchedKITTINormalDataset(Dataset):
    """Split-safe KITTI adapter for official RGB--geometry baselines."""

    def __init__(self, root: Path, split: str, baseline: str, height: int, width: int) -> None:
        self.root = root
        self.split = split
        self.baseline = baseline
        self.height = height
        self.width = width
        self.training = split == "training"
        self.samples = sorted((root / split / "image_2").glob("*.png"))
        if not self.samples:
            raise FileNotFoundError(f"No images under {root / split / 'image_2'}")
        for image_path in self.samples:
            sample = image_path.stem
            geometry_path = (
                root / split / "ADI" / f"{sample}.png"
                if baseline == "plard"
                else root / split / "normal" / f"{sample}.npy"
            )
            required = [geometry_path, self._label_path(sample)]
            if self.training and baseline == "usnet":
                required.append(root / split / "normal_flipped" / f"{sample}.npy")
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing cached inputs for {sample}: {missing}")

    def _label_path(self, sample: str) -> Path:
        category, frame = sample.split("_", 1)
        return self.root / self.split / "gt_image_2" / f"{category}_road_{frame}.png"

    @staticmethod
    def _road_label(path: Path) -> Image.Image:
        label_rgb = np.asarray(Image.open(path).convert("RGB"))
        # Match LiteViLNet's local perspective-view label conversion exactly.
        road = ((label_rgb[:, :, 2] > 200) & (label_rgb[:, :, 0] > 200)).astype(np.uint8)
        return Image.fromarray(road)

    @staticmethod
    def _gaussian_noise(image: Image.Image, sigma: float = 10.0) -> Image.Image:
        array = np.asarray(image).astype(np.int32)
        for channel in range(3):
            noise = np.random.normal(0.0, sigma, array[:, :, channel].size)
            noisy = np.clip(array[:, :, channel].reshape(-1) + noise, 0, 255)
            array[:, :, channel] = noisy.reshape(array.shape[:2])
        return Image.fromarray(array.astype(np.uint8))

    @staticmethod
    def _color_jitter(image: Image.Image) -> Image.Image:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.6, 1.4))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.6, 1.4))
        return ImageEnhance.Color(image).enhance(random.uniform(0.6, 1.4))

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.samples[index]
        sample = image_path.stem
        if self.baseline == "plard":
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            adi = cv2.imread(
                str(self.root / self.split / "ADI" / f"{sample}.png"),
                cv2.IMREAD_UNCHANGED,
            )
            if image is None or adi is None:
                raise RuntimeError(f"Failed to decode PLARD inputs for {sample}")
            if adi.ndim == 3:
                adi = adi[:, :, 0]
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            adi = cv2.resize(adi, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            # Match the official PLARD KITTIRoadLoader exactly: OpenCV BGR,
            # Pascal/PSPNet mean subtraction, and positive-ADI centering.
            rgb = image.astype(np.float32)
            rgb -= np.asarray((103.939, 116.779, 123.68), dtype=np.float32)
            geometry = adi.astype(np.float32) / 128.0
            positives = geometry[geometry > 0]
            if positives.size:
                geometry -= float(positives.mean())
            label = self._road_label(self._label_path(sample)).resize(
                (self.width, self.height), Image.Resampling.NEAREST
            )
            return {
                "rgb": torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))),
                "normal": torch.from_numpy(np.ascontiguousarray(geometry[None, :, :])),
                "label": torch.from_numpy(np.ascontiguousarray(np.asarray(label, dtype=np.int64))),
                "name": sample,
            }

        image = Image.open(image_path).convert("RGB")
        label = self._road_label(self._label_path(sample))

        flipped = self.training and self.baseline == "usnet" and random.random() < 0.5
        normal_dir = "normal_flipped" if flipped else "normal"
        normal = np.load(self.root / self.split / normal_dir / f"{sample}.npy", allow_pickle=False)
        if flipped:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if self.training and self.baseline == "usnet":
            if random.random() < 0.5:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))
            if random.random() < 0.5:
                image = self._gaussian_noise(image)

        image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        label = label.resize((self.width, self.height), Image.Resampling.NEAREST)
        normal = cv2.resize(np.transpose(normal, (1, 2, 0)), (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        if self.training and self.baseline == "usnet":
            image = self._color_jitter(image)

        rgb = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        if self.baseline == "usnet":
            rgb = (rgb - RGB_MEAN) / RGB_STD
        normal = np.asarray(normal, dtype=np.float32).transpose(2, 0, 1)
        target = np.asarray(label, dtype=np.int64)
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb)),
            "normal": torch.from_numpy(np.ascontiguousarray(normal)),
            "label": torch.from_numpy(np.ascontiguousarray(target)),
            "name": sample,
        }

    def __len__(self) -> int:
        return len(self.samples)


@dataclass
class ModelBundle:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    base_lr: float
    official_files: list[Path]


def load_usnet(args: argparse.Namespace, device: torch.device) -> ModelBundle:
    source = str(args.official_source.resolve())
    sys.path.insert(0, source)
    from model.usnet import USNet  # pylint: disable=import-outside-toplevel

    model = USNet(2, "resnet18").to(device)
    base_lr = args.learning_rate if args.learning_rate is not None else 1e-3
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    base_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters},
            {"params": model.backbone.parameters(), "lr": base_lr * 0.1},
        ],
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    return ModelBundle(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        base_lr=base_lr,
        official_files=[
            args.official_source / "model" / "usnet.py",
            args.official_source / "loss.py",
            args.official_source / "train.py",
            args.official_source / "utils.py",
            args.official_source / "dataset" / "kitti.py",
            args.official_source / "dataset" / "custom_transforms.py",
        ],
    )


def load_sne_roadseg(args: argparse.Namespace, device: torch.device) -> ModelBundle:
    source = str(args.official_source.resolve())
    sys.path.insert(0, source)
    from models.networks import define_RoadSeg  # pylint: disable=import-outside-toplevel

    # The official initializer keeps ``need_initialization`` on RoadSeg and
    # reaches it through a DataParallel wrapper.  Supplying the visible CUDA
    # device reproduces that intended path and avoids changing official code.
    gpu_ids = [0] if device.type == "cuda" else []
    model = define_RoadSeg(2, use_sne=True, init_type="kaiming", init_gain=0.02, gpu_ids=gpu_ids).to(device)
    base_lr = args.learning_rate if args.learning_rate is not None else 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 0.9 ** ((epoch + 1) // 25)
    )
    return ModelBundle(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        base_lr=base_lr,
        official_files=[
            args.official_source / "models" / "networks.py",
            args.official_source / "models" / "roadseg_model.py",
            args.official_source / "models" / "sne_model.py",
            args.official_source / "train.py",
            args.official_source / "data" / "kitti_dataset.py",
            args.official_source / "options" / "base_options.py",
            args.official_source / "options" / "train_options.py",
        ],
    )


def load_plard(args: argparse.Namespace, device: torch.device) -> ModelBundle:
    source = str(args.official_source.resolve())
    sys.path.insert(0, source)
    # The pinned repository contains protobuf code generated by an older
    # protoc.  The pure-Python implementation preserves its semantics on
    # modern protobuf releases without modifying the official source tree.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    from ptsemseg.models import get_model  # pylint: disable=import-outside-toplevel

    model = get_model("plard", 2).to(device)
    base_lr = args.learning_rate if args.learning_rate is not None else 1e-4
    optimizer = torch.optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.99, weight_decay=5e-4
    )
    # The paper specifies a gradual 1e-4 -> 1e-6 decay.  Its released CLI
    # defaults contradict the accompanying 30-epoch comment, so the adapter
    # implements the published decay directly over the common epoch budget.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: max(0.01, (1.0 - min(epoch, args.epochs) / args.epochs) ** 0.9),
    )
    return ModelBundle(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        base_lr=base_lr,
        official_files=[
            args.official_source / "ptsemseg" / "models" / "plard.py",
            args.official_source / "ptsemseg" / "models" / "utils.py",
            args.official_source / "ptsemseg" / "loss.py",
            args.official_source / "ptsemseg" / "loader" / "kitti_road_loader.py",
            args.official_source / "train.py",
        ],
    )


def usnet_loss(outputs: tuple[Any, ...], labels: torch.Tensor, epoch: int, official_source: Path) -> torch.Tensor:
    if str(official_source.resolve()) not in sys.path:
        sys.path.insert(0, str(official_source.resolve()))
    from loss import ce_loss  # pylint: disable=import-outside-toplevel

    _, alpha_sup, _, _, alpha, alpha_a = outputs
    target = labels.reshape(-1)
    loss: torch.Tensor | float = 0.0
    # These weights and the 50-epoch KL annealing horizon are the official
    # USNet training objective in train.py.
    for view in range(len(alpha_sup)):
        loss = loss + ce_loss(target, alpha_sup[view].float(), 2, epoch, 50)
    for view in range(len(alpha)):
        loss = loss + ce_loss(target, alpha[view].float(), 2, epoch, 50)
    loss = loss + 2.0 * ce_loss(target, alpha_a.float(), 2, epoch, 50)
    return torch.mean(loss)


def probabilities(baseline: str, model: torch.nn.Module, rgb: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    outputs = model([rgb, normal]) if baseline == "plard" else model(rgb, normal)
    if baseline == "usnet":
        alpha_a = outputs[-1]
        probability = alpha_a / torch.sum(alpha_a, dim=1, keepdim=True)
        return probability[:, 1].reshape(rgb.shape[0], rgb.shape[2], rgb.shape[3])
    if baseline == "plard":
        # Official PLARD eval mode already returns class probabilities.
        return outputs[:, 1]
    return torch.softmax(outputs, dim=1)[:, 1]


@torch.no_grad()
def validate(
    baseline: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    meter = BinarySegmentationMeter(thresholds=101)
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        normal = batch["normal"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            probability = probabilities(baseline, model, rgb, normal)
        meter.update(probability, label, input_type="prob")
    return meter.compute()


def update_usnet_lr(optimizer: torch.optim.Optimizer, base_lr: float, epoch: int, epochs: int) -> float:
    lr = base_lr * (1.0 - epoch / epochs) ** 0.9
    optimizer.param_groups[0]["lr"] = lr
    optimizer.param_groups[1]["lr"] = lr * 0.1
    return lr


def save_checkpoint(
    path: Path,
    bundle: ModelBundle,
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
) -> None:
    state = {
        "model_state_dict": bundle.model.state_dict(),
        "optimizer_state_dict": bundle.optimizer.state_dict(),
        "scheduler_state_dict": bundle.scheduler.state_dict() if bundle.scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    torch.save(state, path)


def main() -> None:
    args = parse_args()
    if args.amp and not args.device.startswith("cuda"):
        raise ValueError("--amp currently requires CUDA")
    source_clean = official_source_is_semantically_clean(args.official_source)
    if not source_clean and not args.allow_modified_official_source:
        raise RuntimeError(
            f"Official source has semantic modifications: {args.official_source}. "
            "Use a clean pinned clone; the override is for diagnostics only."
        )
    source_commit = git_value(args.official_source, ["rev-parse", "HEAD"])
    if source_commit != PINNED_COMMITS[args.baseline] and not args.allow_unpinned_official_source:
        raise RuntimeError(
            f"Expected {args.baseline} commit {PINNED_COMMITS[args.baseline]}, found {source_commit}. "
            "Check out the pinned official commit; the override is for diagnostics only."
        )
    seed_all(args.seed)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. "
            "Use a new directory or pass --resume with its checkpoint."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_set = MatchedKITTINormalDataset(args.data_root, "training", args.baseline, args.height, args.width)
    val_set = MatchedKITTINormalDataset(args.data_root, "validation", args.baseline, args.height, args.width)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed + 1),
        persistent_workers=args.num_workers > 0,
    )

    if args.baseline == "usnet":
        bundle = load_usnet(args, device)
    elif args.baseline == "sne_roadseg":
        bundle = load_sne_roadseg(args, device)
    else:
        bundle = load_plard(args, device)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    start_epoch = 0
    best_metric = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        bundle.model.load_state_dict(checkpoint["model_state_dict"])
        bundle.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if bundle.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            bundle.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_metric = float(checkpoint["best_metric"])

    log_path = args.output_dir / "train_metrics.jsonl"
    stale_validations = 0
    global_step = 0
    last_metrics: dict[str, float] | None = None
    train_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        bundle.model.train()
        if args.baseline == "plard":
            bundle.model.freeze_bn()
        if args.baseline == "usnet":
            lr = update_usnet_lr(bundle.optimizer, bundle.base_lr, epoch, args.epochs)
        else:
            lr = bundle.optimizer.param_groups[0]["lr"]
        epoch_losses: list[float] = []
        epoch_start = time.time()
        for batch_index, batch in enumerate(train_loader):
            rgb = batch["rgb"].to(device, non_blocking=True)
            normal = batch["normal"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            bundle.optimizer.zero_grad(set_to_none=True)
            if args.baseline == "plard":
                # The official PLARD forward owns its three-scale loss and
                # backward call.  Keep that path intact and use FP32, as in
                # the released implementation and paper.
                loss = bundle.model([rgb, normal, label], global_step)
            else:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                    outputs = bundle.model(rgb, normal)
                    if args.baseline == "usnet":
                        loss = usnet_loss(outputs, label, epoch, args.official_source)
                    else:
                        loss = F.cross_entropy(outputs, label)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}, batch {batch_index + 1}: {loss}")
            if args.baseline == "plard":
                bundle.optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(bundle.optimizer)
                scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))
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
            "learning_rate": lr,
            "train_loss": float(np.mean(epoch_losses)),
            "epoch_seconds": time.time() - epoch_start,
            "global_step": global_step,
        }
        if should_validate:
            last_metrics = validate(args.baseline, bundle.model, val_loader, device, args.amp)
            record["validation"] = last_metrics
            current = last_metrics["MaxF"]
            if current > best_metric:
                best_metric = current
                stale_validations = 0
                save_checkpoint(args.output_dir / "best_model.pth", bundle, epoch + 1, best_metric, args)
            else:
                stale_validations += 1
            record["best_MaxF"] = best_metric
            record["stale_validations"] = stale_validations
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

        if args.smoke_steps and global_step >= args.smoke_steps:
            save_checkpoint(args.output_dir / "smoke_model.pth", bundle, epoch + 1, best_metric, args)
            break
        if args.early_stop_validations > 0 and stale_validations >= args.early_stop_validations:
            print(f"Early stopping after {stale_validations} validation checks without MaxF improvement")
            break

    best_path = args.output_dir / "best_model.pth"
    if best_path.is_file():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        bundle.model.load_state_dict(checkpoint["model_state_dict"])
        final_metrics = validate(args.baseline, bundle.model, val_loader, device, args.amp)
        best_epoch = int(checkpoint["epoch"])
    else:
        final_metrics = last_metrics or validate(args.baseline, bundle.model, val_loader, device, args.amp)
        best_epoch = epoch + 1

    source_remote = git_value(args.official_source, ["remote", "get-url", "origin"])
    result = {
        "protocol": "matched local KITTI perspective-view retraining",
        "baseline": args.baseline,
        "seed": args.seed,
        "input_size": [args.height, args.width],
        "train_count": len(train_set),
        "val_count": len(val_set),
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "amp": args.amp,
        "metrics": final_metrics,
        "parameters": sum(parameter.numel() for parameter in bundle.model.parameters()),
        "best_checkpoint": str(best_path.resolve()) if best_path.is_file() else None,
        "best_checkpoint_sha256": sha256(best_path) if best_path.is_file() else None,
        "official_source": str(args.official_source.resolve()),
        "official_remote": source_remote,
        "official_commit": source_commit,
        "official_semantic_diff_clean": source_clean,
        "official_file_sha256": {str(path.relative_to(args.official_source)): sha256(path) for path in bundle.official_files},
        "normal_cache_metadata": (
            json.loads((args.data_root / "normal_cache_metadata.json").read_text(encoding="utf-8"))
            if args.baseline != "plard"
            else None
        ),
        "split_metadata": json.loads((args.data_root / "matched_split_metadata.json").read_text(encoding="utf-8")),
        "runtime": {
            "seconds": time.time() - train_start,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "command": sys.argv,
        },
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
