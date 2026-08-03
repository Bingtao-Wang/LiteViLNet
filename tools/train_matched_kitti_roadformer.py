#!/usr/bin/env python3
"""Retrain the official RoadFormer graph under the matched KITTI protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from litevilnet.metrics.deployment_metrics import BinarySegmentationMeter


PINNED_COMMIT = "f675a3467cb168ebc727648390c304279bbcb079"
EXPECTED_REMOTE = "https://github.com/LiJiahang617/Road-Former.git"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--early-stop-validations", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-steps", type=int, default=0)
    return parser.parse_args()


def git_value(path: Path, arguments: list[str]) -> str:
    return subprocess.check_output(["git", *arguments], cwd=path, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def bootstrap_official(source: Path) -> dict[str, Any]:
    """Register official RoadFormer while reusing an installed MMCV ABI."""

    sys.path.insert(0, str(source.resolve()))
    import mmcv._ext as mmcv_ext  # pylint: disable=import-outside-toplevel
    import mmcv.ops as mmcv_ops  # pylint: disable=import-outside-toplevel

    sys.modules["mmcv_custom._ext"] = mmcv_ext
    runtime_ops = types.ModuleType("mmcv_custom.ops")
    for name in (
        "batched_nms",
        "point_sample",
        "MultiScaleDeformableAttention",
        "sigmoid_focal_loss",
        "nms_match",
        "nms",
    ):
        setattr(runtime_ops, name, getattr(mmcv_ops, name))
    runtime_ops.__path__ = []
    sys.modules["mmcv_custom.ops"] = runtime_ops
    for name in ("multi_scale_deform_attn", "roi_align"):
        sys.modules[f"mmcv_custom.ops.{name}"] = importlib.import_module(f"mmcv.ops.{name}")

    from mmseg_custom.utils import register_all_modules  # pylint: disable=import-outside-toplevel

    register_all_modules()
    return {
        "strategy": "official Python graph with installed MMCV 1.7 CUDA operators mapped at runtime",
        "mapped_symbols": [
            "batched_nms",
            "point_sample",
            "MultiScaleDeformableAttention",
            "sigmoid_focal_loss",
            "nms_match",
            "nms",
            "roi_align",
        ],
    }


def dataset_config(root: Path, split: str, width: int, height: int, training: bool) -> dict[str, Any]:
    pipeline: list[dict[str, Any]] = [
        dict(type="LoadKittiImageFromFile", to_float32=True, modality="normal"),
        dict(type="StackByChannel", keys=("img", "ano")),
        dict(type="LoadKittiAnnotations", reduce_zero_label=False),
        dict(type="Resize", scale=(width, height)),
    ]
    if training:
        pipeline.extend(
            [
                dict(type="RandomFlip", prob=0.5),
                dict(
                    type="MultiModalPhotoMetricDistortion",
                    brightness_delta=24,
                    contrast_range=(0.7, 1.3),
                    saturation_range=(0.7, 1.3),
                    hue_delta=12,
                ),
            ]
        )
    pipeline.append(dict(type="PackSegInputs"))
    return dict(
        type="MMKittiDataset",
        data_root=str(root.resolve()),
        reduce_zero_label=False,
        img_suffix=".png",
        modality="normal",
        data_prefix=dict(
            img_path=f"image_2/{split}",
            normal_path=f"sne/{split}",
            seg_map_path=f"gt_image_2/{split}",
        ),
        pipeline=pipeline,
    )


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader) -> dict[str, float]:
    model.eval()
    meter = BinarySegmentationMeter(thresholds=101)
    for batch in loader:
        predictions = model.val_step(batch)
        for prediction, target_sample in zip(predictions, batch["data_samples"]):
            logits = prediction.seg_logits.data.float()
            probability = torch.softmax(logits, dim=0)[1]
            target = target_sample.gt_sem_seg.data.squeeze(0)
            # Official post-processing restores logits to the raw KITTI
            # dimensions, whereas the common evaluator uses the resized
            # network target.  Explicit batch dimensions let the shared
            # meter perform the same bilinear alignment used for all rows.
            meter.update(probability.unsqueeze(0), target.unsqueeze(0), input_type="prob")
    return meter.compute()


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: Any, epoch: int, metric: float) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": metric,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("RoadFormer matched training requires CUDA")
    source = args.official_source.resolve()
    if git_value(source, ["rev-parse", "HEAD"]) != PINNED_COMMIT:
        raise RuntimeError(f"RoadFormer source must be pinned at {PINNED_COMMIT}")
    remote = git_value(source, ["remote", "get-url", "origin"])
    if remote != EXPECTED_REMOTE:
        raise RuntimeError(f"Unexpected RoadFormer origin: {remote}")
    if git_value(source, ["status", "--porcelain"]):
        raise RuntimeError("RoadFormer official source is not clean")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    operator_bridge = bootstrap_official(source)

    from mmengine_custom.config import Config  # pylint: disable=import-outside-toplevel
    from mmengine_custom.dataset import pseudo_collate  # pylint: disable=import-outside-toplevel
    from mmengine_custom.optim import build_optim_wrapper  # pylint: disable=import-outside-toplevel
    from mmseg_custom.registry import DATASETS, MODELS  # pylint: disable=import-outside-toplevel

    config_path = source / "configs" / "roadformer_kitti" / "roadformer_convnext-b_kitti-384x1280.py"
    config = Config.fromfile(config_path)
    config.model.data_preprocessor.size = (args.height, args.width)
    config.model.decode_head.pixel_decoder.img_scale = (args.height, args.width)
    if args.amp:
        config.optim_wrapper.type = "AmpOptimWrapper"
        config.optim_wrapper.loss_scale = "dynamic"
    train_set = DATASETS.build(dataset_config(args.data_root, "training", args.width, args.height, True))
    val_set = DATASETS.build(dataset_config(args.data_root, "validation", args.width, args.height, False))
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pseudo_collate,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pseudo_collate,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed + 1),
        persistent_workers=args.num_workers > 0,
    )

    model = MODELS.build(config.model).cuda()
    model.init_weights()
    optim_wrapper = build_optim_wrapper(model, config.optim_wrapper)
    initial_lrs = [group["lr"] for group in optim_wrapper.optimizer.param_groups]
    start_epoch = 0
    best_metric = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cuda")
        model.load_state_dict(checkpoint["model_state_dict"])
        optim_wrapper.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_metric = float(checkpoint["best_metric"])

    train_start = time.time()
    stale_validations = 0
    global_step = 0
    log_path = args.output_dir / "train_metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        progress = epoch / max(args.epochs - 1, 1)
        for initial_lr, group in zip(initial_lrs, optim_wrapper.optimizer.param_groups):
            eta_min = min(1e-5, initial_lr)
            group["lr"] = eta_min + (initial_lr - eta_min) * (1.0 - progress) ** 0.9
        model.train()
        losses: list[float] = []
        epoch_start = time.time()
        for batch in train_loader:
            log_vars = model.train_step(batch, optim_wrapper)
            loss = float(log_vars["loss"].detach().cpu())
            if not np.isfinite(loss):
                raise FloatingPointError(f"Non-finite RoadFormer loss: {loss}")
            losses.append(loss)
            global_step += 1
            if args.smoke_steps and global_step >= args.smoke_steps:
                break
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "learning_rates": sorted(
                {round(float(group["lr"]), 12) for group in optim_wrapper.optimizer.param_groups}
            ),
            "train_loss": float(np.mean(losses)),
            "epoch_seconds": time.time() - epoch_start,
            "global_step": global_step,
        }
        should_validate = args.smoke_steps > 0 or (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs
        if should_validate:
            metrics = validate(model, val_loader)
            record["validation"] = metrics
            if metrics["MaxF"] > best_metric:
                best_metric = metrics["MaxF"]
                stale_validations = 0
                save_checkpoint(args.output_dir / "best_model.pth", model, optim_wrapper, epoch + 1, best_metric)
            else:
                stale_validations += 1
            record["best_MaxF"] = best_metric
            record["stale_validations"] = stale_validations
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if args.smoke_steps and global_step >= args.smoke_steps:
            break
        if args.early_stop_validations > 0 and stale_validations >= args.early_stop_validations:
            break

    best_path = args.output_dir / "best_model.pth"
    checkpoint = torch.load(best_path, map_location="cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = validate(model, val_loader)
    split_metadata = json.loads((args.data_root / "matched_split_metadata.json").read_text())
    official_files = [
        config_path,
        source / "mmpretrain_custom" / "models" / "backbones" / "twin_convnext.py",
        source / "mmdet_custom" / "models" / "dense_heads" / "roadformer_head.py",
        source / "mmdet_custom" / "models" / "layers" / "roadformer_pixel_decoder.py",
        source / "mmseg_custom" / "models" / "decode_heads" / "roadformer_head.py",
    ]
    result = {
        "protocol": "matched local KITTI perspective-view retraining",
        "baseline": "roadformer",
        "seed": args.seed,
        "input_size": [args.height, args.width],
        "train_count": len(train_set),
        "val_count": len(val_set),
        "epochs_requested": args.epochs,
        "best_epoch": int(checkpoint["epoch"]),
        "batch_size": args.batch_size,
        "amp": args.amp,
        "metrics": final_metrics,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256(best_path),
        "official_source": str(source),
        "official_remote": remote,
        "official_commit": PINNED_COMMIT,
        "official_semantic_diff_clean": True,
        "official_file_sha256": {
            str(path.relative_to(source)): sha256(path) for path in official_files
        },
        "operator_bridge": operator_bridge,
        "split_metadata": split_metadata,
        "runtime": {
            "seconds": time.time() - train_start,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "command": sys.argv,
        },
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
