"""Generate RGB-only road pseudo-labels for collected robot images.

The outputs are intended as annotation pre-labels, not ground-truth labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

from litevilnet.utils.common import system_metadata


MODEL_REGISTRY = {
    "segformer_cityscapes": {
        "model_id": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        "local_dir": "other_models/segformer_cityscapes",
        "backend": "semantic",
        "road_labels": ("road",),
    },
    "mask2former_cityscapes": {
        "model_id": "facebook/mask2former-swin-base-IN21k-cityscapes-semantic",
        "local_dir": "other_models/mask2former_cityscapes",
        "backend": "mask2former",
        "road_labels": ("road",),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict RGB road pseudo-labels")
    parser.add_argument("--input_dir", default="data/reality")
    parser.add_argument("--output_dir", default="output/2.0_reality_rgb_pseudolabel_compare")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY), choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for semantic logits")
    parser.add_argument("--opacity", type=float, default=0.45)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models_root", default="other_models")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def image_paths(input_dir: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in suffixes)


def label_ids(model_config: Any, names: tuple[str, ...]) -> list[int]:
    id2label = getattr(model_config, "id2label", {})
    matched = []
    wanted = {name.lower() for name in names}
    for raw_idx, raw_label in id2label.items():
        label = str(raw_label).lower().replace("_", " ").strip()
        if label in wanted:
            matched.append(int(raw_idx))
    if not matched:
        raise ValueError(f"Could not find labels {names!r} in model id2label={id2label!r}")
    return matched


def resize_prob(prob: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    resized = F.interpolate(
        prob.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    return resized.detach().cpu().numpy()


def predict_semantic(
    image: Image.Image,
    processor: AutoImageProcessor,
    model: AutoModelForSemanticSegmentation,
    road_ids: list[int],
    device: torch.device,
) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits.float()
    probs = torch.softmax(logits, dim=1)
    road_prob = probs[:, road_ids].sum(dim=1).squeeze(0)
    return resize_prob(road_prob, image.size)


def predict_mask2former(
    image: Image.Image,
    processor: Mask2FormerImageProcessor,
    model: Mask2FormerForUniversalSegmentation,
    road_ids: list[int],
    device: torch.device,
) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    processed = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    mask = torch.zeros_like(processed, dtype=torch.float32)
    for road_id in road_ids:
        mask = torch.maximum(mask, (processed == road_id).float())
    return mask.cpu().numpy()


def make_overlay(image: Image.Image, prob: np.ndarray, threshold: float, opacity: float) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    alpha = (prob >= threshold).astype(np.float32) * opacity
    color = np.zeros_like(base)
    color[..., 1] = 255.0
    blended = base * (1.0 - alpha[..., None]) + color * alpha[..., None]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def write_outputs(
    model_name: str,
    model_id: str,
    paths: list[Path],
    probs: list[np.ndarray],
    output_dir: Path,
    threshold: float,
    opacity: float,
) -> list[dict[str, Any]]:
    root = output_dir / model_name
    mask_dir = root / "mask"
    prob_dir = root / "prob"
    overlay_dir = root / "overlay"
    for directory in (mask_dir, prob_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for path, prob in zip(paths, probs):
        image = Image.open(path).convert("RGB")
        mask = (prob >= threshold).astype(np.uint8) * 255
        prob_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
        stem = path.stem
        mask_path = mask_dir / f"{stem}_road_mask.png"
        prob_path = prob_dir / f"{stem}_road_prob.png"
        overlay_path = overlay_dir / f"{stem}_road_overlay.jpg"
        Image.fromarray(mask, mode="L").save(mask_path)
        Image.fromarray(prob_u8, mode="L").save(prob_path)
        make_overlay(image, prob, threshold, opacity).save(overlay_path, quality=95)
        rows.append(
            {
                "input": str(path),
                "mask": str(mask_path),
                "prob": str(prob_path),
                "overlay": str(overlay_path),
                "model": model_name,
                "model_id": model_id,
                "original_size": [image.size[0], image.size[1]],
                "positive_ratio": float((prob >= threshold).mean()),
                "prob_mean": float(prob.mean()),
                "prob_max": float(prob.max()),
            }
        )
    return rows


def run_model(model_name: str, paths: list[Path], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    spec = MODEL_REGISTRY[model_name]
    model_id = spec["model_id"]
    model_source = Path(spec["local_dir"])
    if not model_source.exists():
        model_source = Path(args.models_root) / model_name
    if model_source.exists():
        model_source_arg = str(model_source)
    elif args.local_files_only:
        raise FileNotFoundError(
            f"Local model directory not found for {model_name}: {model_source}. "
            "Run tools/download_rgb_pseudolabel_models.py first."
        )
    else:
        model_source_arg = model_id
    backend = spec["backend"]
    device = torch.device(args.device)

    if backend == "semantic":
        processor = AutoImageProcessor.from_pretrained(model_source_arg, local_files_only=args.local_files_only)
        model = AutoModelForSemanticSegmentation.from_pretrained(
            model_source_arg,
            local_files_only=args.local_files_only,
        ).to(device).eval()
        road_ids = label_ids(model.config, spec["road_labels"])
        predict = lambda image: predict_semantic(image, processor, model, road_ids, device)
    elif backend == "mask2former":
        processor = Mask2FormerImageProcessor.from_pretrained(model_source_arg, local_files_only=args.local_files_only)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_source_arg,
            local_files_only=args.local_files_only,
        ).to(device).eval()
        road_ids = label_ids(model.config, spec["road_labels"])
        predict = lambda image: predict_mask2former(image, processor, model, road_ids, device)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    probs = []
    with torch.no_grad():
        for path in tqdm(paths, desc=f"Predicting {model_name}"):
            image = Image.open(path).convert("RGB")
            probs.append(predict(image))

    files = write_outputs(model_name, model_id, paths, probs, output_dir, args.threshold, args.opacity)
    return {
        "name": model_name,
        "model_id": model_id,
        "model_source": model_source_arg,
        "backend": backend,
        "road_ids": road_ids,
        "files": files,
    }


def write_readme(output_dir: Path, args: argparse.Namespace) -> None:
    (output_dir / "README.md").write_text(
        "# 2.0_reality_rgb_pseudolabel_compare\n\n"
        "## Purpose\n\n"
        "RGB-only road pseudo-label generation for robot/phone-captured images.\n\n"
        "## Important Limitation\n\n"
        "These outputs are automatic pre-labels. They are not ground truth until a human reviews and fixes them.\n\n"
        "## Inputs\n\n"
        f"- Input directory: `{args.input_dir}`\n"
        f"- Threshold: `{args.threshold}`\n\n"
        "## Outputs\n\n"
        "Each model folder contains `mask/`, `prob/`, and `overlay/`.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = image_paths(input_dir)
    if not paths:
        raise FileNotFoundError(f"No RGB images found under {input_dir}")

    results = []
    errors = []
    for model_name in args.models:
        try:
            results.append(run_model(model_name, paths, output_dir, args))
        except Exception as exc:
            errors.append({"name": model_name, "error": repr(exc)})
            print(f"[ERROR] {model_name}: {exc}")

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "threshold": args.threshold,
        "opacity": args.opacity,
        "models": args.models,
        "count": len(paths),
        "results": results,
        "errors": errors,
        "note": "Automatic RGB-only pseudo-labels. Human review is required before using them as GT.",
        "system": system_metadata(),
    }
    write_readme(output_dir, args)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "count": len(paths), "errors": errors}, indent=2, ensure_ascii=False))
    if len(results) == 0:
        raise RuntimeError("All pseudo-label models failed.")


if __name__ == "__main__":
    main()
