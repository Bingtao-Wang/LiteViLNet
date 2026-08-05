#!/usr/bin/env python3
"""Retrain the pinned official RoadFormer graph on released ORFD splits."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from litevilnet.metrics.deployment_metrics import BinarySegmentationMeter
from tools.train_matched_kitti_roadformer import (
    EXPECTED_REMOTE,
    PINNED_COMMIT,
    bootstrap_official,
    git_value,
    seed_all,
    seed_worker,
    sha256,
)
from tools.train_matched_orfd_baseline import fixed_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--early-stop-validations", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--smoke-eval-samples", type=int, default=8)
    return parser.parse_args()


def dataset_config(
    root: Path, split: str, width: int, height: int, training: bool
) -> dict[str, Any]:
    pipeline: list[dict[str, Any]] = [
        dict(type="LoadOrfdImageFromFile", to_float32=True, modality="normal"),
        dict(type="StackByChannel", keys=("img", "ano")),
    ]
    if training:
        pipeline.extend(
            [
                dict(type="LoadOrfdAnnotations", reduce_zero_label=False),
                dict(type="Resize", scale=(width, height)),
            ]
        )
    else:
        # This ordering is the official RoadFormer ORFD test pipeline.
        pipeline.extend(
            [
                dict(type="Resize", scale=(width, height)),
                dict(type="LoadOrfdAnnotations", reduce_zero_label=False),
            ]
        )
    pipeline.append(dict(type="PackSegInputs"))
    return dict(
        type="MMOrfdDataset",
        data_root=str(root.resolve()),
        reduce_zero_label=False,
        img_suffix=".png",
        seg_map_suffix="_fillcolor.png",
        modality="normal",
        data_prefix=dict(
            img_path=f"images/{split}",
            normal_path=f"normal/{split}",
            seg_map_path=f"annotations/{split}",
        ),
        pipeline=pipeline,
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, max_samples: int = 0) -> dict[str, Any]:
    model.eval()
    meter = BinarySegmentationMeter(thresholds=101)
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    processed = 0
    threshold_swept_grid: list[int] | None = None
    fixed_argmax_ground_truth_grid: list[int] | None = None
    for batch in loader:
        # Use pre-postprocess logits on the common network-input grid. The
        # released ``val_step`` bilinearly restores RoadFormer logits to raw
        # metadata before argmax, while the other compared methods threshold
        # their native/input-grid outputs and then restore a discrete mask.
        data = model.data_preprocessor(batch, False)
        data_samples = data["data_samples"]
        batch_img_metas = [sample.metainfo for sample in data_samples]
        batch_logits = model.encode_decode(data["inputs"], batch_img_metas).float()
        for logits, target_sample in zip(batch_logits, data_samples):
            probability = torch.softmax(logits, dim=0)[1]
            threshold_swept_grid = [int(value) for value in probability.shape[-2:]]
            target_original = target_sample.gt_sem_seg.data.squeeze(0).to(
                probability.device
            ).bool()
            fixed_argmax_ground_truth_grid = [
                int(value) for value in target_original.shape[-2:]
            ]
            target_at_input = torch.nn.functional.interpolate(
                target_original.float()[None, None],
                size=probability.shape[-2:],
                mode="nearest",
            )[0, 0].bool()
            meter.update(
                probability.unsqueeze(0), target_at_input.unsqueeze(0), input_type="prob"
            )
            prediction_mask = probability > 0.5
            if prediction_mask.shape != target_original.shape:
                prediction_mask = torch.nn.functional.interpolate(
                    prediction_mask.float()[None, None],
                    size=target_original.shape,
                    mode="nearest",
                )[0, 0].bool()
            counts["tp"] += int((prediction_mask & target_original).sum().item())
            counts["fp"] += int((prediction_mask & ~target_original).sum().item())
            counts["tn"] += int((~prediction_mask & ~target_original).sum().item())
            counts["fn"] += int((~prediction_mask & target_original).sum().item())
            processed += 1
            if max_samples and processed >= max_samples:
                break
        if max_samples and processed >= max_samples:
            break
    return {
        "samples": processed,
        "threshold_swept_grid": threshold_swept_grid,
        "fixed_argmax_ground_truth_grid": fixed_argmax_ground_truth_grid,
        "threshold_swept": meter.compute(),
        "official_fixed_argmax": fixed_metrics(counts),
    }


def save_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: Any, epoch: int, metric: float
) -> None:
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
        raise RuntimeError("RoadFormer ORFD training requires CUDA")
    source = args.official_source.resolve()
    commit = git_value(source, ["rev-parse", "HEAD"])
    remote = git_value(source, ["remote", "get-url", "origin"])
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"RoadFormer source must be pinned at {PINNED_COMMIT}")
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

    config_path = source / "configs" / "roadformer_orfd" / "roadformer_convnext-b_orfd-352x640.py"
    config = Config.fromfile(config_path)
    config.model.data_preprocessor.size = (args.height, args.width)
    config.model.decode_head.pixel_decoder.img_scale = (args.height, args.width)
    datasets = {
        split: DATASETS.build(
            dataset_config(
                args.data_root,
                released_split,
                args.width,
                args.height,
                split == "train",
            )
        )
        for split, released_split in (
            ("train", "training"),
            ("val", "validation"),
            ("test", "testing"),
        )
    }
    train_loader = DataLoader(
        datasets["train"],
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
    eval_loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=pseudo_collate,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(args.seed + offset),
            persistent_workers=args.num_workers > 0,
        )
        for split, offset in (("val", 1), ("test", 2))
    }

    model = MODELS.build(config.model).cuda()
    model.init_weights()
    optim_wrapper = build_optim_wrapper(model, config.optim_wrapper)
    torch.cuda.reset_peak_memory_stats()
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
            eta_min = min(1e-4, initial_lr)
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
        should_validate = (
            args.smoke_steps > 0
            or (epoch + 1) % args.val_every == 0
            or epoch + 1 == args.epochs
        )
        if should_validate:
            limit = args.smoke_eval_samples if args.smoke_steps else 0
            validation = evaluate(model, eval_loaders["val"], limit)
            record["validation"] = validation
            current = float(validation["threshold_swept"]["MaxF"])
            if current > best_metric:
                best_metric = current
                stale_validations = 0
                save_checkpoint(
                    args.output_dir / "best_model.pth", model, optim_wrapper, epoch + 1, best_metric
                )
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
    checkpoint = torch.load(best_path, map_location="cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    limit = args.smoke_eval_samples if args.smoke_steps else 0
    validation = evaluate(model, eval_loaders["val"], limit)
    testing = evaluate(model, eval_loaders["test"], limit)
    split_metadata = json.loads(
        (args.data_root / "matched_split_metadata.json").read_text(encoding="utf-8")
    )
    official_files = [
        config_path,
        source / "configs" / "_base_" / "datasets" / "mmorfd_640x352.py",
        source / "mmcv_custom" / "transforms" / "loading.py",
        source / "mmseg_custom" / "datasets" / "mmorfd.py",
        source / "mmpretrain_custom" / "models" / "backbones" / "twin_convnext.py",
        source / "mmdet_custom" / "models" / "dense_heads" / "roadformer_head.py",
        source / "mmdet_custom" / "models" / "layers" / "roadformer_pixel_decoder.py",
    ]
    result = {
        "protocol": "local ORFD released-split retraining",
        "baseline": "roadformer",
        "seed": args.seed,
        "input_size": [args.height, args.width],
        "train_count": len(datasets["train"]),
        "val_count": len(datasets["val"]),
        "test_count": len(datasets["test"]),
        "epochs_requested": args.epochs,
        "best_epoch": int(checkpoint["epoch"]),
        "batch_size": args.batch_size,
        "amp": False,
        "checkpoint_selection": "validation 101-threshold MaxF",
        "validation": validation,
        "testing": testing,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256(best_path),
        "official_source": str(source),
        "official_remote": remote,
        "official_commit": commit,
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
            "peak_allocated_cuda_mib": torch.cuda.max_memory_allocated() / (1024**2),
        },
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
