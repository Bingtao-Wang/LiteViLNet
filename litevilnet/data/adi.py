"""Vectorized reimplementation of the released PLARD KITTI ADI procedure."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


def read_kitti_calibration(path: str | Path) -> np.ndarray:
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        try:
            values[key] = np.asarray([float(value) for value in raw.split()], dtype=np.float64)
        except ValueError:
            continue
    p2 = values["P2"].reshape(3, 4)
    r_rect = np.eye(4, dtype=np.float64)
    r_rect[:3, :3] = values["R0_rect"].reshape(3, 3)
    velo_to_cam = np.eye(4, dtype=np.float64)
    velo_to_cam[:3, :] = values["Tr_velo_to_cam"].reshape(3, 4)
    return p2 @ r_rect @ velo_to_cam


def project_sparse_lidar_height(
    points: np.ndarray,
    projection: np.ndarray,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project points exactly as the released MATLAB ADI code specifies."""
    points = np.asarray(points, dtype=np.float64)
    points = points[points[:, 0] >= 5.0, :4].copy()
    points[:, 3] = 1.0
    projected = (projection @ points.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    # MATLAB round() on these positive image coordinates is floor(x + 0.5).
    uv = np.floor(uv + 0.5).astype(np.int64)
    inside = (
        (uv[:, 0] > 0)
        & (uv[:, 0] <= image_width)
        & (uv[:, 1] > 0)
        & (uv[:, 1] <= image_height)
    )
    uv = uv[inside]
    points = points[inside]
    ranges = np.linalg.norm(points[:, :3], axis=1)
    order = np.argsort(ranges, kind="stable")
    # Preserve the original MATLAB one-based pixel convention, then convert
    # the retained indices to zero-based NumPy locations.
    flat = (uv[order, 1] - 1) * image_width + (uv[order, 0] - 1)
    _, first = np.unique(flat, return_index=True)
    selected = order[first]
    rows = uv[selected, 1] - 1
    columns = uv[selected, 0] - 1
    sparse_height = np.zeros((image_height, image_width), dtype=np.float32)
    valid = np.zeros((image_height, image_width), dtype=np.float32)
    sparse_height[rows, columns] = points[selected, 2].astype(np.float32)
    valid[rows, columns] = 1.0
    return sparse_height, valid


def interpolate_height_21x21(
    sparse_height: np.ndarray,
    sparse_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.arange(-10, 11, dtype=np.float32)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    weights = 1.0 / np.maximum(np.sqrt(xx * xx + yy * yy), 1.0)
    numerator = cv2.filter2D(
        sparse_height,
        cv2.CV_32F,
        weights,
        borderType=cv2.BORDER_CONSTANT,
    )
    denominator = cv2.filter2D(
        sparse_valid,
        cv2.CV_32F,
        weights,
        borderType=cv2.BORDER_CONSTANT,
    )
    dense_height = np.zeros_like(sparse_height, dtype=np.float32)
    interpolated = denominator > 0
    dense_height[interpolated] = numerator[interpolated] / denominator[interpolated]
    observed = sparse_valid > 0
    dense_height[observed] = sparse_height[observed]
    dense_valid = (observed | interpolated).astype(np.float32)
    return dense_height, dense_valid


def altitude_gradient_7x7(dense_height: np.ndarray, dense_valid: np.ndarray) -> np.ndarray:
    """Average the released PLARD altitude-gradient response over 7x7 neighbors.

    Slice views avoid constructing two full-frame shifted arrays for every one
    of the 49 offsets.  The coefficient below is algebraically identical to
    ``sqrt((difference/dx)^2 + (difference/dy)^2)`` with each zero-denominator
    term defined as zero by the reference implementation.
    """
    valid = dense_valid > 0
    altitude_sum = np.zeros_like(dense_height, dtype=np.float32)
    counts = np.zeros_like(dense_height, dtype=np.float32)
    height, width = dense_height.shape
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            center_height = dense_height[y0:y1, x0:x1]
            neighbor_height = dense_height[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            pair_valid = valid[y0:y1, x0:x1] & valid[
                y0 + dy : y1 + dy, x0 + dx : x1 + dx
            ]
            coefficient = np.sqrt(
                (1.0 / (dx * dx) if dx != 0 else 0.0)
                + (1.0 / (dy * dy) if dy != 0 else 0.0)
            )
            magnitude = np.abs(neighbor_height - center_height) * coefficient
            altitude_view = altitude_sum[y0:y1, x0:x1]
            count_view = counts[y0:y1, x0:x1]
            altitude_view[pair_valid] += magnitude[pair_valid]
            count_view[pair_valid] += 1.0
    output = np.zeros_like(dense_height, dtype=np.float32)
    nonempty = counts > 0
    output[nonempty] = altitude_sum[nonempty] / counts[nonempty]
    return output


def normalize_adi(raw_adi: np.ndarray) -> np.ndarray:
    scaled = np.zeros_like(raw_adi, dtype=np.float32)
    positive = raw_adi > 0
    if np.any(positive):
        scaled[positive] = (raw_adi[positive] - raw_adi[positive].min()) * 20.0
    scaled = np.sqrt(scaled)
    # MATLAB imgaussfilt(x, 2) defaults to a 9x9 filter and replicate padding.
    smoothed = cv2.GaussianBlur(scaled, (9, 9), sigmaX=2.0, borderType=cv2.BORDER_REPLICATE)
    return np.clip(smoothed, 0.0, 1.0)


def generate_adi(
    points: np.ndarray,
    projection: np.ndarray,
    image_height: int,
    image_width: int,
    return_timings: bool = False,
):
    timings = {}
    start = time.perf_counter()
    sparse_height, sparse_valid = project_sparse_lidar_height(
        points, projection, image_height, image_width
    )
    timings["projection_ms"] = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    dense_height, dense_valid = interpolate_height_21x21(sparse_height, sparse_valid)
    timings["interpolation_ms"] = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    raw_adi = altitude_gradient_7x7(dense_height, dense_valid)
    adi = normalize_adi(raw_adi)
    timings["gradient_and_normalization_ms"] = (time.perf_counter() - start) * 1000.0
    timings["adi_total_ms"] = sum(timings.values())
    return (adi, timings) if return_timings else adi
