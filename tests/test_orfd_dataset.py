import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from litevilnet.data.orfd_dataset import ORFDDataset, _load_image_strict


class ORFDDatasetTest(unittest.TestCase):
    def test_strict_image_loader_retries_a_transient_short_read(self):
        buffer = io.BytesIO()
        Image.fromarray(np.asarray([[17]], dtype=np.uint8)).save(buffer, format="PNG")
        path = Path("transient-network-image.png")
        with (
            patch.object(Path, "read_bytes", side_effect=[b"not a png", buffer.getvalue()]),
            patch("litevilnet.data.orfd_dataset.time.sleep"),
        ):
            image = _load_image_strict(path, attempts=2)
        self.assertEqual(np.asarray(image).tolist(), [[17]])

    def test_strict_image_loader_names_a_persistently_bad_file(self):
        path = Path("persistently-bad.png")
        with (
            patch.object(Path, "read_bytes", return_value=b"not a png"),
            patch("litevilnet.data.orfd_dataset.time.sleep"),
        ):
            with self.assertRaisesRegex(OSError, r"persistently-bad\.png.*5 attempts"):
                _load_image_strict(path)

    def test_direct_archive_layout_depth3_and_grayscale_label(self):
        with tempfile.TemporaryDirectory() as directory:
            split_root = Path(directory) / "training"
            for modality in ("image_data", "dense_depth", "gt_image"):
                (split_root / modality).mkdir(parents=True, exist_ok=True)

            name = "1234567890"
            rgb = np.asarray(
                [
                    [[255, 0, 0], [0, 255, 0]],
                    [[0, 0, 255], [255, 255, 255]],
                ],
                dtype=np.uint8,
            )
            depth = np.asarray([[0, 65535], [32768, 1]], dtype=np.uint16)
            label = np.asarray([[0, 255], [127, 201]], dtype=np.uint8)
            Image.fromarray(rgb).save(split_root / "image_data" / f"{name}.png")
            Image.fromarray(depth).save(split_root / "dense_depth" / f"{name}.png")
            Image.fromarray(label).save(
                split_root / "gt_image" / f"{name}_fillcolor.png"
            )

            dataset = ORFDDataset(
                directory,
                split="train",
                img_h=2,
                img_w=2,
                use_augmentation=False,
            )
            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            self.assertEqual(tuple(sample["rgb"].shape), (3, 2, 2))
            self.assertEqual(tuple(sample["adi"].shape), (3, 2, 2))
            self.assertEqual(tuple(sample["label"].shape), (2, 2))
            self.assertEqual(sample["label"].tolist(), [[0, 1], [0, 1]])

            depth_norm = depth.astype(np.float32) / 65535.0
            valid = depth > 0
            expected = np.stack(
                [depth_norm, valid.astype(np.float32), valid * (1.0 - depth_norm)],
                axis=0,
            )
            expected = (expected - 0.5) / 0.5
            np.testing.assert_allclose(sample["adi"].numpy(), expected, atol=1e-6)

    def test_depth_resize_matches_official_float32_bilinear_path(self):
        depth = np.asarray([[0, 65535], [32768, 0]], dtype=np.uint16)
        actual = ORFDDataset._depth3_tensor(Image.fromarray(depth), img_h=3, img_w=3)

        depth_norm = cv2.resize(
            depth.astype(np.float32) / 65535.0,
            (3, 3),
            interpolation=cv2.INTER_LINEAR,
        )
        valid = depth_norm > 0.0
        expected = np.stack(
            [depth_norm, valid.astype(np.float32), valid * (1.0 - depth_norm)],
            axis=0,
        )
        expected = (expected - 0.5) / 0.5
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6)

    def test_released_test_label_is_loaded_when_present(self):
        with tempfile.TemporaryDirectory() as directory:
            split_root = Path(directory) / "testing"
            for modality in ("image_data", "dense_depth", "gt_image"):
                (split_root / modality).mkdir(parents=True, exist_ok=True)

            name = "test_frame"
            Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(
                split_root / "image_data" / f"{name}.png"
            )
            Image.fromarray(np.ones((2, 2), dtype=np.uint16)).save(
                split_root / "dense_depth" / f"{name}.png"
            )
            label = np.asarray([[0, 255], [255, 0]], dtype=np.uint8)
            Image.fromarray(label).save(
                split_root / "gt_image" / f"{name}_fillcolor.png"
            )

            sample = ORFDDataset(
                directory,
                split="test",
                img_h=2,
                img_w=2,
                use_augmentation=False,
                return_original_label=True,
            )[0]
            self.assertEqual(sample["label"].tolist(), [[0, 1], [1, 0]])
            self.assertEqual(sample["label_original"].tolist(), [[0, 1], [1, 0]])


if __name__ == "__main__":
    unittest.main()
