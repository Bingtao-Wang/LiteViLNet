#!/usr/bin/env python
"""Export KITTI Road prediction masks for server/devkit evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from litevilnet.model_factory import MODEL_PRESETS, available_presets, build_model
from litevilnet.utils.common import system_metadata, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export KITTI Road prediction masks")
    parser.add_argument("--preset", default="litevilnet_baseline", choices=available_presets())
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data_root", default="data/kitti_road")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--category", default="all")
    parser.add_argument("--img_h", type=int, default=384)
    parser.add_argument("--img_w", type=int, default=1248)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--allow_partial", action="store_true", help="Allow non-aux partial checkpoint loading")
    parser.add_argument("--allow_random_init", action="store_true", help="Allow export without a checkpoint for smoke tests")
    parser.add_argument("--save_prob", action="store_true", help="Also save uint8 probability maps under prob/")
    return parser.parse_args()


def submission_name(image_name: str) -> str:
    stem = Path(image_name).stem
    if "_" not in stem:
        raise ValueError(f"Cannot derive KITTI Road submission name from {image_name!r}")
    prefix, suffix = stem.split("_", 1)
    return f"{prefix}_road_{suffix}.png"


class KITTIPredictionDataset(Dataset):
    """KITTI Road images for prediction export."""

    def __init__(self, data_root: str, split: str, category: str, img_h: int, img_w: int):
        self.data_root = Path(data_root)
        self.split = split
        self.category = category
        self.img_h = img_h
        self.img_w = img_w
        self.samples = self._collect_samples()
        if not self.samples:
            raise FileNotFoundError(
                f"No samples found for split={split!r}, category={category!r} under {self.data_root}. "
                "Expected training/image_2 + training/ADI for train/val or testing/image_2 + testing/ADI for test."
            )

    def _collect_samples(self) -> list[dict[str, Path | str]]:
        if self.split == "test":
            image_dir = self.data_root / "testing" / "image_2"
            adi_dir = self.data_root / "testing" / "ADI"
        else:
            image_dir = self.data_root / "training" / "image_2"
            adi_dir = self.data_root / "training" / "ADI"

        if not image_dir.exists() or not adi_dir.exists():
            return []

        samples = []
        for image_path in sorted(image_dir.glob("*.png")):
            prefix = image_path.name.split("_", 1)[0]
            if self.category != "all" and prefix != self.category:
                continue
            adi_path = adi_dir / image_path.name
            if not adi_path.exists():
                continue
            samples.append({"name": image_path.name, "rgb": image_path, "adi": adi_path})

        if self.split == "train":
            samples = samples[: int(len(samples) * 0.8)]
        elif self.split == "val":
            samples = samples[int(len(samples) * 0.8) :]
        return samples

    @staticmethod
    def _load_image(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
        image = Image.open(path).convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0, image.size

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray((image * 255.0).astype(np.uint8))
        resized = pil.resize((self.img_w, self.img_h), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        array = ((array - mean) / std).transpose(2, 0, 1)
        return torch.from_numpy(array)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | tuple[int, int]]:
        sample = self.samples[idx]
        rgb, original_size = self._load_image(sample["rgb"])
        adi, _ = self._load_image(sample["adi"])
        return {
            "rgb": self._preprocess(rgb),
            "adi": self._preprocess(adi),
            "name": sample["name"],
            "original_size": original_size,
        }


def _resize_batch(probs: torch.Tensor, sizes: list[tuple[int, int]]) -> list[np.ndarray]:
    masks = []
    for prob, size in zip(probs, sizes):
        width, height = size
        resized = F.interpolate(
            prob.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        masks.append(resized.detach().cpu().numpy())
    return masks


def collate_prediction_batch(batch: list[dict[str, torch.Tensor | str | tuple[int, int]]]) -> dict[str, object]:
    return {
        "rgb": torch.stack([item["rgb"] for item in batch]),
        "adi": torch.stack([item["adi"] for item in batch]),
        "name": [item["name"] for item in batch],
        "original_size": [item["original_size"] for item in batch],
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_arg = args.checkpoint or MODEL_PRESETS[args.preset].get("checkpoint_hint", "")
    if checkpoint_arg and not Path(checkpoint_arg).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_arg}")
    if not checkpoint_arg and not args.allow_random_init:
        raise ValueError("A checkpoint is required for KITTI export. Use --allow_random_init only for smoke tests.")
    checkpoint = checkpoint_arg or None
    model, metadata = build_model(
        args.preset,
        checkpoint=checkpoint,
        device=device,
        strict=False,
        allow_partial=args.allow_partial,
    )
    if args.precision == "fp16":
        model.half()

    dataset = KITTIPredictionDataset(args.data_root, args.split, args.category, args.img_h, args.img_w)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_prediction_batch,
    )

    output_dir = Path(args.output_dir or f"runs/kitti_submission/{args.preset}")
    mask_dir = output_dir
    mask_dir.mkdir(parents=True, exist_ok=True)
    prob_dir = output_dir / "prob"
    if args.save_prob:
        prob_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Exporting {metadata['label']}"):
            rgb = batch["rgb"].to(device, non_blocking=True)
            adi = batch["adi"].to(device, non_blocking=True)
            if args.precision == "fp16":
                rgb = rgb.half()
                adi = adi.half()
            logits = model(rgb, adi)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits.squeeze(1).float())
            resized_probs = _resize_batch(probs, batch["original_size"])

            for image_name, prob in zip(batch["name"], resized_probs):
                out_name = submission_name(image_name)
                mask = (prob > args.threshold).astype(np.uint8) * 255
                Image.fromarray(mask, mode="L").save(mask_dir / out_name)
                if args.save_prob:
                    prob_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
                    Image.fromarray(prob_u8, mode="L").save(prob_dir / out_name)
                exported.append(out_name)

    manifest = {
        "preset": args.preset,
        "label": metadata["label"],
        "checkpoint": checkpoint,
        "split": args.split,
        "category": args.category,
        "threshold": args.threshold,
        "input_shape": [args.batch_size, 3, args.img_h, args.img_w],
        "output_dir": str(output_dir),
        "mask_dir": str(mask_dir),
        "count": len(exported),
        "files": exported,
        "system": system_metadata(),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
