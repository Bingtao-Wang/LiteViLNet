#!/usr/bin/env python
"""Create MP4 videos from robot prediction image sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make robot prediction videos")
    parser.add_argument("--frames_dir", default="")
    parser.add_argument("--masks_dir", default="")
    parser.add_argument("--panels_dir", default="", help="Directory containing rgb/depth_color/prediction_overlay/probability panels")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def write_video(image_paths: list[Path], output_path: Path, fps: float, is_mask: bool = False) -> None:
    if not image_paths:
        raise FileNotFoundError(f"No frames for {output_path}")
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(image_paths[0])
    if is_mask:
        first = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    for path in image_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(path)
        if is_mask:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR)
        writer.write(frame)
    writer.release()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.panels_dir:
        panels_dir = Path(args.panels_dir)
        panel_names = ("rgb", "depth_color", "prediction_overlay", "probability")
        for name in panel_names:
            image_paths = sorted((panels_dir / name).glob("*.jpg"))
            write_video(image_paths, output_dir / f"{name}.mp4", args.fps, is_mask=False)
        return

    if not args.frames_dir or not args.masks_dir:
        raise ValueError("Either --panels_dir or both --frames_dir and --masks_dir must be provided")

    overlay_paths = sorted(Path(args.frames_dir).glob("*.jpg"))
    mask_paths = sorted(Path(args.masks_dir).glob("*.png"))
    write_video(overlay_paths, output_dir / "overlay.mp4", args.fps, is_mask=False)
    write_video(mask_paths, output_dir / "mask.mp4", args.fps, is_mask=True)


if __name__ == "__main__":
    main()
