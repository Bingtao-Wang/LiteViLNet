"""Robot-road RGB+aligned-depth dataset and explicit depth3 encoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def encode_depth3(depth_mm: np.ndarray, max_depth_mm: float = 12000.0) -> np.ndarray:
    """Encode aligned millimeter depth as depth, validity, and inverse depth."""
    depth = depth_mm.astype(np.float32)
    valid = (depth > 0.0) & (depth < max_depth_mm)
    depth_norm = np.clip(depth, 0.0, max_depth_mm) / max_depth_mm
    valid_f = valid.astype(np.float32)
    inverse_depth = valid_f * (1.0 - depth_norm)
    depth3 = np.stack([depth_norm, valid_f, inverse_depth], axis=-1)
    return (depth3 - 0.5) / 0.5


def load_rgb_tensor(path: Path, img_h: int, img_w: int) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    original_size = image.size
    image = image.resize((img_w, img_h), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = ((array - RGB_MEAN) / RGB_STD).transpose(2, 0, 1)
    return torch.from_numpy(array.copy()), original_size


def load_depth3_tensor(
    path: Path,
    img_h: int,
    img_w: int,
    max_depth_mm: float = 12000.0,
) -> torch.Tensor:
    depth = Image.open(path).resize((img_w, img_h), Image.Resampling.NEAREST)
    depth3 = encode_depth3(np.asarray(depth), max_depth_mm=max_depth_mm).transpose(2, 0, 1)
    return torch.from_numpy(depth3.astype(np.float32, copy=True))


def load_mask_tensor(path: Path, img_h: int, img_w: int) -> torch.Tensor:
    mask = Image.open(path).convert("L").resize((img_w, img_h), Image.Resampling.NEAREST)
    return torch.from_numpy((np.asarray(mask) > 0).astype(np.int64).copy())


def _session_dir(data_root: Path, session: str) -> Path:
    nested = data_root / "raw" / session
    direct = data_root / session
    return nested if nested.exists() else direct


class RobotRoadRGBDepthDataset(Dataset):
    """RGB + aligned depth + binary walkable-path masks."""

    def __init__(
        self,
        data_root: str | Path,
        session: str,
        split_file: str | Path,
        img_h: int = 384,
        img_w: int = 608,
        max_depth_mm: float = 12000.0,
        require_mask: bool = True,
        mask_dir: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.session = session
        self.img_h = img_h
        self.img_w = img_w
        self.max_depth_mm = max_depth_mm
        self.require_mask = require_mask
        session_dir = _session_dir(self.data_root, session)
        self.rgb_dir = session_dir / "rgb"
        self.depth_dir = session_dir / "depth"
        self.mask_dir = Path(mask_dir) if mask_dir else self.data_root / "annotations" / "manual_masks"
        frames = [
            line.strip()
            for line in Path(split_file).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        if not frames:
            raise ValueError(f"No frames listed in split file: {split_file}")
        self.samples = []
        for frame in frames:
            rgb_path = self.rgb_dir / f"{frame}.png"
            depth_path = self.depth_dir / f"{frame}.png"
            mask_path = self.mask_dir / f"{frame}.png"
            required = [rgb_path, depth_path] + ([mask_path] if require_mask else [])
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Missing robot-road files for frame {frame}: {missing}")
            self.samples.append({"frame": frame, "rgb": rgb_path, "depth": depth_path, "mask": mask_path})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        rgb, original_size = load_rgb_tensor(sample["rgb"], self.img_h, self.img_w)
        item = {
            "rgb": rgb,
            "depth3": load_depth3_tensor(sample["depth"], self.img_h, self.img_w, self.max_depth_mm),
            "name": sample["frame"],
            "original_size": original_size,
        }
        if self.require_mask:
            item["label"] = load_mask_tensor(sample["mask"], self.img_h, self.img_w)
        return item


class RobotRoadPredictionDataset(Dataset):
    """All paired RGB+depth frames from one robot session."""

    def __init__(
        self,
        data_root: str | Path,
        session: str,
        img_h: int = 384,
        img_w: int = 608,
        max_depth_mm: float = 12000.0,
    ) -> None:
        self.data_root = Path(data_root)
        self.session = session
        self.img_h = img_h
        self.img_w = img_w
        self.max_depth_mm = max_depth_mm
        session_dir = _session_dir(self.data_root, session)
        self.samples = []
        for rgb_path in sorted((session_dir / "rgb").glob("*.png")):
            depth_path = session_dir / "depth" / rgb_path.name
            if depth_path.exists():
                self.samples.append({"frame": rgb_path.stem, "rgb": rgb_path, "depth": depth_path})
        if not self.samples:
            raise FileNotFoundError(f"No paired RGB+depth frames below {session_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        rgb, original_size = load_rgb_tensor(sample["rgb"], self.img_h, self.img_w)
        return {
            "rgb": rgb,
            "depth3": load_depth3_tensor(sample["depth"], self.img_h, self.img_w, self.max_depth_mm),
            "name": sample["frame"],
            "original_size": original_size,
        }


def resize_probs_to_original(
    probabilities: torch.Tensor,
    sizes: list[tuple[int, int]],
) -> list[np.ndarray]:
    """Resize BxHxW probabilities to each original (width, height)."""
    resized = []
    for probability, (width, height) in zip(probabilities, sizes):
        output = F.interpolate(
            probability[None, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        resized.append(output.detach().cpu().numpy())
    return resized
