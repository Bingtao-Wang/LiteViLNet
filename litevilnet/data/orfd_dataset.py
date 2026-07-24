"""Official-split ORFD RGB+dense-depth freespace dataset adapter."""

from __future__ import annotations

import io
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset


ORFD_SPLITS = {"train": "training", "val": "validation", "test": "testing"}
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMAGE_READ_ATTEMPTS = 5


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _load_image_strict(path: Path, attempts: int = IMAGE_READ_ATTEMPTS) -> Image.Image:
    """Read and fully decode an image, retrying transient network-file errors.

    ORFD is large enough that it is commonly kept on network storage.  PIL
    opens files lazily, so a short-lived CIFS interruption can otherwise
    surface later as ``broken data stream`` without naming the offending
    path.  Reading into memory and calling ``load`` makes each attempt atomic
    from the dataset's perspective.  Truncated-image decoding is deliberately
    not enabled: a persistently damaged file still fails loudly.
    """

    if attempts < 1:
        raise ValueError("attempts must be positive")
    last_error: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            payload = path.read_bytes()
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                return image.copy()
        except OSError as error:
            last_error = error
            if attempt < attempts:
                time.sleep(0.1 * (2 ** (attempt - 1)))
    raise OSError(f"Failed to read and decode {path} after {attempts} attempts") from last_error


class ORFDDataset(Dataset):
    """Load ORFD using its released training/validation/testing partitions.

    The second stream is not called ADI: ORFD supplies a registered dense-depth
    image.  We encode it as ``[normalized depth, valid mask, inverse depth]``
    and map the three channels to [-1, 1].  This is the same explicit depth3
    interface used by LiteViLNet's RGB-D deployment branch.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        img_h: int = 352,
        img_w: int = 640,
        use_augmentation: bool = True,
        return_original_label: bool = False,
    ) -> None:
        if split not in ORFD_SPLITS:
            raise ValueError(f"Unknown ORFD split {split!r}; choose from {sorted(ORFD_SPLITS)}")
        self.data_root = Path(data_root)
        self.split = split
        self.img_h = img_h
        self.img_w = img_w
        self.use_augmentation = use_augmentation and split == "train"
        self.return_original_label = return_original_label
        split_root = self.data_root / ORFD_SPLITS[split]
        self.samples = []
        # The 2022 Google-Drive archive stores each modality directly below
        # the split, while the current official README also documents an
        # optional sequence subdirectory. Support both released layouts.
        rgb_paths = list(split_root.glob("image_data/*.png"))
        rgb_paths.extend(split_root.glob("*/image_data/*.png"))
        for rgb_path in sorted(set(rgb_paths)):
            sequence_root = rgb_path.parent.parent
            depth_path = sequence_root / "dense_depth" / rgb_path.name
            label_path = sequence_root / "gt_image" / f"{rgb_path.stem}_fillcolor.png"
            if depth_path.is_file() and (split == "test" or label_path.is_file()):
                self.samples.append(
                    {
                        "name": f"{sequence_root.name}/{rgb_path.stem}",
                        "rgb": rgb_path,
                        "depth": depth_path,
                        "label": label_path,
                    }
                )
        if not self.samples:
            raise FileNotFoundError(
                f"No ORFD samples found below {split_root}; expected [*/]image_data/*.png "
                "with matching dense_depth and gt_image files"
            )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _rgb_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        return (tensor - RGB_MEAN) / RGB_STD

    @staticmethod
    def _depth3_tensor(depth_image: Image.Image, img_h: int, img_w: int) -> torch.Tensor:
        depth = np.asarray(depth_image, dtype=np.float32)
        # The released ORFD loader treats dense_depth as uint16 and divides by
        # 65535, then applies OpenCV's default bilinear resize. Preserve those
        # released operations before extending the result to depth3.
        depth_norm = np.clip(depth, 0.0, 65535.0) / 65535.0
        depth_norm = cv2.resize(
            depth_norm,
            (img_w, img_h),
            interpolation=cv2.INTER_LINEAR,
        )
        valid = (depth_norm > 0.0) & (depth_norm <= 1.0)
        inverse = valid.astype(np.float32) * (1.0 - depth_norm)
        depth3 = np.stack([depth_norm, valid.astype(np.float32), inverse], axis=0)
        return torch.from_numpy(depth3.copy()).float().sub_(0.5).div_(0.5)

    @staticmethod
    def _label_tensor(label_image: Image.Image) -> torch.Tensor:
        label = np.asarray(label_image.convert("RGB"))
        # Match the official ORFD loader: after BGR->RGB conversion it selects
        # channel index 2 > 200, i.e. the blue channel in RGB order.
        return torch.from_numpy((label[:, :, 2] > 200).astype(np.int64).copy())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        rgb = _load_image_strict(sample["rgb"]).convert("RGB")
        depth = _load_image_strict(sample["depth"])
        if sample["label"].is_file():
            label = _load_image_strict(sample["label"]).convert("RGB")
        else:
            # Keep prediction-only compatibility with ORFD releases that do
            # not bundle testing labels.  The 2022 archive used in the paper
            # does include gt_image for testing, so those labels are loaded
            # above for the held-out benchmark.
            label = Image.new("RGB", rgb.size)

        if self.use_augmentation and random.random() < 0.5:
            rgb = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.use_augmentation and random.random() < 0.3:
            rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(0.8, 1.2))
            rgb = ImageEnhance.Contrast(rgb).enhance(random.uniform(0.8, 1.2))

        label_original = self._label_tensor(label)
        size = (self.img_w, self.img_h)
        rgb = rgb.resize(size, Image.Resampling.BILINEAR)
        label = label.resize(size, Image.Resampling.NEAREST)
        result = {
            "rgb": self._rgb_tensor(rgb),
            "adi": self._depth3_tensor(depth, self.img_h, self.img_w),
            "label": self._label_tensor(label),
            "name": sample["name"],
        }
        if self.return_original_label:
            result["label_original"] = label_original
        return result


def get_orfd_dataloader(
    data_root: str | Path,
    split: str,
    batch_size: int,
    num_workers: int,
    img_h: int,
    img_w: int,
    use_augmentation: bool,
    seed: int,
    drop_last: bool = False,
    shuffle: bool | None = None,
    return_original_label: bool = False,
) -> DataLoader:
    dataset = ORFDDataset(
        data_root=data_root,
        split=split,
        img_h=img_h,
        img_w=img_w,
        use_augmentation=use_augmentation,
        return_original_label=return_original_label,
    )
    if shuffle is None:
        shuffle = split == "train"
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
