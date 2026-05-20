#!/usr/bin/env python
"""Predict all robot RGB+Depth frames with LiteViLNetRGBDepth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from litevilnet.data.robot_road_dataset import RobotRoadPredictionDataset, resize_probs_to_original
from litevilnet.models.litevilnet_rgbdepth import LiteViLNetRGBDepth
from litevilnet.utils.common import system_metadata, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict robot walkable path masks")
    parser.add_argument("--data_root", default="data/robot_road")
    parser.add_argument("--session", default="session_20260509_172746")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics", default="")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--img_h", type=int, default=384)
    parser.add_argument("--img_w", type=int, default=608)
    parser.add_argument("--max_depth_mm", type=float, default=12000.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames_file", default="", help="Optional newline-separated frame list to predict")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def collate_prediction_batch(batch):
    return {
        "rgb": torch.stack([item["rgb"] for item in batch]),
        "depth3": torch.stack([item["depth3"] for item in batch]),
        "name": [item["name"] for item in batch],
        "original_size": [item["original_size"] for item in batch],
    }


def threshold_from_metrics(metrics_path: str) -> float:
    if not metrics_path or not Path(metrics_path).exists():
        return 0.5
    data = json.loads(Path(metrics_path).read_text())
    candidates = []
    if isinstance(data.get("latest"), dict):
        candidates.append(data["latest"].get("BestThreshold"))
    if isinstance(data.get("history"), list) and data["history"]:
        best = max(data["history"], key=lambda row: row.get("MaxF", -1.0))
        candidates.append(best.get("BestThreshold"))
    for value in candidates:
        if value is not None:
            return float(value)
    return 0.5


def load_model(checkpoint_path: str, device: torch.device) -> LiteViLNetRGBDepth:
    model = LiteViLNetRGBDepth(pretrained=False, use_deep_supervision=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def draw_panel_title(image: np.ndarray, title: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(image, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)


def depth_to_vis(depth_path: Path, shape: tuple[int, int], max_depth_mm: float) -> np.ndarray:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(depth_path)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.shape[:2] != shape:
        depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    depth_f = depth.astype(np.float32)
    valid = (depth_f > 0.0) & (depth_f < max_depth_mm)
    normalized = np.zeros_like(depth_f, dtype=np.uint8)
    if valid.any():
        valid_depth = depth_f[valid]
        low, high = np.percentile(valid_depth, [2.0, 98.0])
        if high <= low:
            low, high = float(valid_depth.min()), float(valid_depth.max())
        if high > low:
            normalized[valid] = np.clip((valid_depth - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    depth_vis[~valid] = 0

    valid_ratio = float(valid.mean() * 100.0)
    if valid.any():
        valid_depth_m = depth_f[valid] / 1000.0
        stats = (
            f"Depth valid {valid_ratio:.1f}%  "
            f"vis {low / 1000.0:.2f}-{high / 1000.0:.2f}m  "
            f"min/med/max {valid_depth_m.min():.2f}/{np.median(valid_depth_m):.2f}/{valid_depth_m.max():.2f}m"
        )
    else:
        stats = "Depth valid 0.0%  no valid depth"
    draw_panel_title(depth_vis, stats)
    return depth_vis


def save_overlay(
    rgb_path: Path,
    depth_path: Path,
    prob: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    max_depth_mm: float,
    panels_dir: Path | None = None,
) -> None:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    frame = rgb_path.stem
    color = np.zeros_like(rgb)
    color[:, :, 1] = 255
    overlay = np.where(mask[..., None] > 0, (rgb * 0.65 + color * 0.35).astype(np.uint8), rgb)
    depth_vis = depth_to_vis(depth_path, rgb.shape[:2], max_depth_mm)
    heat = cv2.applyColorMap(np.clip(prob * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    draw_panel_title(rgb, "RGB")
    draw_panel_title(overlay, "Prediction overlay")
    draw_panel_title(heat, "Walkable probability")
    if panels_dir is not None:
        panel_images = {
            "rgb": rgb,
            "depth_color": depth_vis,
            "prediction_overlay": overlay,
            "probability": heat,
        }
        for name, image in panel_images.items():
            directory = panels_dir / name
            directory.mkdir(parents=True, exist_ok=True)
            suffix = ".jpg" if name in {"rgb", "depth_color", "prediction_overlay", "probability"} else ".png"
            cv2.imwrite(str(directory / f"{frame}{suffix}"), image)

    top = np.concatenate([rgb, depth_vis], axis=1)
    bottom = np.concatenate([overlay, heat], axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    cv2.imwrite(str(output_path), grid)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = args.threshold if args.threshold is not None else threshold_from_metrics(args.metrics)

    dataset = RobotRoadPredictionDataset(
        data_root=args.data_root,
        session=args.session,
        img_h=args.img_h,
        img_w=args.img_w,
        max_depth_mm=args.max_depth_mm,
    )
    if args.frames_file:
        requested = {line.strip() for line in Path(args.frames_file).read_text().splitlines() if line.strip()}
        dataset.samples = [sample for sample in dataset.samples if sample["frame"] in requested]
        if not dataset.samples:
            raise ValueError(f"No prediction frames matched --frames_file: {args.frames_file}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_prediction_batch,
    )
    model = load_model(args.checkpoint, device)

    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "mask"
    prob_dir = output_dir / "prob"
    overlay_dir = output_dir / "overlay"
    panels_dir = output_dir / "panels"
    for directory in (mask_dir, prob_dir, overlay_dir, panels_dir):
        directory.mkdir(parents=True, exist_ok=True)

    session_dir = Path(args.data_root) / "raw" / args.session
    if not session_dir.exists():
        session_dir = Path(args.data_root) / args.session
    exported = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predict robot RGB+Depth"):
            rgb = batch["rgb"].to(device, non_blocking=True)
            depth3 = batch["depth3"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp and torch.cuda.is_available()):
                logits = model(rgb, depth3)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits.squeeze(1).float())
            resized_probs = resize_probs_to_original(probs, batch["original_size"])

            for frame, prob in zip(batch["name"], resized_probs):
                mask = (prob > threshold).astype(np.uint8) * 255
                Image.fromarray(mask, mode="L").save(mask_dir / f"{frame}.png")
                Image.fromarray(np.clip(prob * 255.0, 0, 255).astype(np.uint8), mode="L").save(prob_dir / f"{frame}.png")
                save_overlay(
                    session_dir / "rgb" / f"{frame}.png",
                    session_dir / "depth" / f"{frame}.png",
                    prob,
                    mask,
                    overlay_dir / f"{frame}.jpg",
                    args.max_depth_mm,
                    panels_dir,
                )
                exported.append(frame)

    write_json(
        output_dir / "manifest.json",
        {
            "checkpoint": args.checkpoint,
            "metrics": args.metrics,
            "frames_file": args.frames_file,
            "threshold": threshold,
            "count": len(exported),
            "frames": exported,
            "input_size": [args.img_h, args.img_w],
            "system": system_metadata(),
        },
    )


if __name__ == "__main__":
    main()
