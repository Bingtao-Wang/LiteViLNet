#!/usr/bin/env python
"""Generate deterministic reference material for manual RA-L figure revision.

This tool closes four figure-provenance gaps in one reproducible workflow:

* draw the architecture from the implemented encoder/fusion/decoder contract;
* draw MSFM from the tensor operations in ``attention_modules.py``;
* repair the swapped Unitree-G1 depth/segmentation panels; and
* render KITTI qualitative results from the fixed validation manifest and
  the seed-42 full-model checkpoint at its validation-global threshold.

Outputs are written to a reference-material directory and never replace the
author's manuscript figures. The architecture/MSFM composites are layout
references only; qualitative panels and the robot correction copy are source
material for manual drawing. A JSON manifest records inputs and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from litevilnet.data.dataset import KITTIRoadDataset
from litevilnet.models.vllinet_ablation import get_ablation_model


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_FIGURE_DIR = REPO_ROOT / "runs/revision_1/figure_materials"
DEFAULT_SOURCE_FIGURE_DIR = WORKSPACE_ROOT / "LiteViLNetPaperRAL" / "figures"
DEFAULT_DATA_ROOT = Path(os.environ.get("LITEVILNET_KITTI_ROOT", "data/kitti_road"))
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "LITEVILNET_KITTI_FIGURE_CHECKPOINT",
        "runs/revision_1/checkpoints/full_seed42_best_model.pth",
    )
)
DEFAULT_SPLIT = REPO_ROOT / (
    "configs/splits/kitti_road/stratified_seed20260723/val.txt"
)
DEFAULT_MANIFEST = REPO_ROOT / "runs/revision_1/revision_figure_manifest.json"
DEFAULT_SAMPLE_IDS = (
    "um_000054",
    "umm_000017",
    "umm_000050",
    "uu_000059",
)

COLORS = {
    "rgb": "#1677b8",
    "geometry": "#f28e2b",
    "fusion": "#8f63b8",
    "decoder": "#3ba272",
    "bridge": "#d84a4a",
    "neutral": "#425466",
    "light": "#f4f7fa",
    "line": "#334155",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    face: str,
    edge: str | None = None,
    fontsize: float = 8.0,
    weight: str = "normal",
    text_color: str = "#17202a",
    linewidth: float = 1.2,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.25,rounding_size=0.9",
        facecolor=face,
        edgecolor=edge or COLORS["line"],
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=text_color,
        zorder=zorder + 1,
        linespacing=1.15,
    )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    width: float = 1.2,
    style: str = "-|>",
    connection: str = "arc3",
    mutation_scale: float = 10,
    zorder: int = 1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=width,
            color=color,
            connectionstyle=connection,
            shrinkA=1.2,
            shrinkB=1.2,
            zorder=zorder,
        )
    )


def feature_stack(ax, x: float, y: float, color: str, *, scale: float = 1.0) -> None:
    """Draw a compact three-plane feature-map glyph."""
    for offset, alpha in ((1.5, 0.35), (0.75, 0.55), (0.0, 0.85)):
        ax.add_patch(
            Rectangle(
                (x + offset * scale, y + offset * scale),
                4.2 * scale,
                4.6 * scale,
                facecolor=color,
                edgecolor="white",
                linewidth=0.65,
                alpha=alpha,
                zorder=3,
            )
        )


def save_figure(fig, png_path: Path, pdf_path: Path, *, dpi: int = 300) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        metadata={
            "Title": pdf_path.stem,
            "Author": "LiteViLNet revision figure generator",
            "Creator": "tools.generate_revision_figures",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def generate_architecture(figure_dir: Path) -> list[Path]:
    """Draw Fig. 1 from the current LiteViLNet implementation contract."""
    fig, ax = plt.subplots(figsize=(18.0, 7.2))
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 72)
    ax.axis("off")

    ax.text(
        2,
        69.7,
        "KITTI geometry preprocessing (offline, before the network)",
        fontsize=11,
        fontweight="bold",
        color=COLORS["geometry"],
        va="top",
    )
    pre_x = [4, 34, 65, 96, 127, 158]
    pre_text = [
        "LiDAR point cloud",
        "Calibrated\nprojection",
        "21×21 height\ninterpolation",
        "7×7 altitude\ngradient",
        "Per-image normalization\n+ Gaussian σ=2",
        "3-channel ADI",
    ]
    pre_width = [22, 20, 21, 20, 24, 18]
    for i, (x, label, w) in enumerate(zip(pre_x, pre_text, pre_width)):
        rounded_box(
            ax,
            x,
            59.5,
            w,
            7.0,
            label,
            face="#fff2e4" if i else "#f8fafc",
            edge=COLORS["geometry"],
            fontsize=8.2,
            weight="bold" if i in (0, 5) else "normal",
        )
        if i:
            arrow(
                ax,
                (pre_x[i - 1] + pre_width[i - 1], 63.0),
                (x, 63.0),
                color=COLORS["geometry"],
            )

    ax.text(
        2,
        54.6,
        "Dual-stream encoder and scale-wise fusion",
        fontsize=11,
        fontweight="bold",
        color=COLORS["neutral"],
    )

    rounded_box(
        ax,
        3,
        43.0,
        24,
        8.0,
        "RGB image\nH×W×3",
        face="#e4f3ff",
        edge=COLORS["rgb"],
        fontsize=9,
        weight="bold",
    )
    rounded_box(
        ax,
        3,
        23.0,
        24,
        8.0,
        "Precomputed ADI\nH×W×3 (pipeline output)",
        face="#fff2e4",
        edge=COLORS["geometry"],
        fontsize=9,
        weight="bold",
    )

    stage_x = [39, 63, 87, 111, 135]
    channels = [16, 24, 40, 112, 960]
    scales = ["1/2", "1/4", "1/8", "1/16", "1/32"]
    rgb_names = ["MobileNetV3"] * 5
    geo_names = ["Conv stem", "DSConv", "DSConv", "DSConv", "DSConv"]

    arrow(ax, (27, 47), (39, 47), color=COLORS["rgb"], width=1.4)
    arrow(ax, (27, 27), (39, 27), color=COLORS["geometry"], width=1.4)
    for i, x in enumerate(stage_x):
        rounded_box(
            ax,
            x,
            43.0,
            15,
            8.0,
            f"Stage {i + 1}: {rgb_names[i]}\nRes {scales[i]}  •  C={channels[i]}",
            face="#dff1ff",
            edge=COLORS["rgb"],
            fontsize=7.5,
            weight="bold" if i == 0 else "normal",
        )
        rounded_box(
            ax,
            x,
            23.0,
            15,
            8.0,
            f"Stage {i + 1}: {geo_names[i]}\nRes {scales[i]}  •  C={channels[i]}",
            face="#ffedd9",
            edge=COLORS["geometry"],
            fontsize=7.5,
            weight="bold" if i == 0 else "normal",
        )
        rounded_box(
            ax,
            x + 3.0,
            34.0,
            9.0,
            6.0,
            "MSFM\n1×N score",
            face="#efe4fa",
            edge=COLORS["fusion"],
            fontsize=7.2,
            weight="bold",
        )
        arrow(ax, (x + 7.5, 43.0), (x + 7.5, 40.0), color=COLORS["rgb"])
        arrow(ax, (x + 7.5, 31.0), (x + 7.5, 34.0), color=COLORS["geometry"])
        if i < 4:
            arrow(ax, (x + 15, 47.0), (stage_x[i + 1], 47.0), color=COLORS["rgb"])
            arrow(ax, (x + 15, 27.0), (stage_x[i + 1], 27.0), color=COLORS["geometry"])

    rounded_box(
        ax,
        155,
        33.0,
        20,
        8.0,
        "Large-kernel bridge\n1×1 → DWConv 7×7 → 1×1",
        face="#ffe3e3",
        edge=COLORS["bridge"],
        fontsize=7.8,
        weight="bold",
    )
    arrow(ax, (147, 37.0), (155, 37.0), color=COLORS["fusion"], width=1.4)

    ax.text(
        2,
        18.3,
        "U-Net decoder with fused skip features",
        fontsize=11,
        fontweight="bold",
        color=COLORS["decoder"],
    )
    decoder_x = [153, 129, 105, 81, 57]
    decoder_labels = [
        "Bottleneck\n960 → 128",
        "Up block\n1/16, C=64",
        "Up block\n1/8, C=32",
        "Up block\n1/4, C=16",
        "Up block\n1/2, C=16",
    ]
    for i, (x, label) in enumerate(zip(decoder_x, decoder_labels)):
        rounded_box(
            ax,
            x,
            7.0,
            16,
            8.0,
            label,
            face="#ddf5e9",
            edge=COLORS["decoder"],
            fontsize=7.8,
            weight="bold" if i == 0 else "normal",
        )
        if i:
            arrow(
                ax,
                (decoder_x[i - 1], 11.0),
                (x + 16, 11.0),
                color=COLORS["decoder"],
                width=1.4,
            )
    arrow(
        ax,
        (165, 33.0),
        (161, 15.0),
        color=COLORS["bridge"],
        connection="angle3,angleA=-90,angleB=0",
        width=1.4,
    )

    # Fused stage-4 to stage-1 features feed decoder skip connections.
    skip_targets = [137, 113, 89, 65]
    for source_x, target_x in zip(stage_x[3::-1], skip_targets):
        arrow(
            ax,
            (source_x + 12.0, 37.0),
            (target_x, 15.0),
            color=COLORS["fusion"],
            width=1.0,
            connection="arc3,rad=0.24",
            mutation_scale=8,
        )

    rounded_box(
        ax,
        31,
        7.0,
        18,
        8.0,
        "1×1 head + sigmoid\nwalkable-area mask",
        face="#e6f7ef",
        edge=COLORS["decoder"],
        fontsize=8,
        weight="bold",
    )
    arrow(ax, (57, 11.0), (49, 11.0), color=COLORS["decoder"], width=1.4)
    rounded_box(
        ax,
        3,
        7.0,
        20,
        8.0,
        "Segmentation\nH×W",
        face="#f5fff9",
        edge=COLORS["decoder"],
        fontsize=9,
        weight="bold",
    )
    arrow(ax, (31, 11.0), (23, 11.0), color=COLORS["decoder"], width=1.4)

    ax.text(
        178,
        1.0,
        "RGB-D deployment uses a separate aligned-depth → depth3 input path.",
        fontsize=8.5,
        color=COLORS["neutral"],
        ha="right",
        va="bottom",
        style="italic",
    )

    png = figure_dir / "fig_architecture2.png"
    pdf = figure_dir / "fig_architecture.pdf"
    save_figure(fig, png, pdf, dpi=300)
    return [png, pdf]


def generate_msfm(figure_dir: Path) -> list[Path]:
    """Draw Fig. 3 using the implemented global-query MSFM operations."""
    fig, ax = plt.subplots(figsize=(17.5, 7.0))
    ax.set_xlim(0, 175)
    ax.set_ylim(0, 70)
    ax.axis("off")

    ax.text(
        2,
        68.5,
        "MSFM at one scale l",
        fontsize=13,
        fontweight="bold",
        color=COLORS["neutral"],
        va="top",
    )
    ax.text(
        173,
        68.5,
        "Global RGB query • spatial geometry keys/values • linear 1×Nₗ score memory",
        fontsize=9.5,
        color=COLORS["fusion"],
        va="top",
        ha="right",
        fontweight="bold",
    )

    rounded_box(
        ax, 2, 46, 18, 9, "RGB feature\nB×Cₗ×Hₗ×Wₗ", face="#dff1ff", edge=COLORS["rgb"], fontsize=8.5, weight="bold"
    )
    feature_stack(ax, 7.5, 56.0, COLORS["rgb"], scale=0.85)
    rounded_box(
        ax, 2, 12, 18, 9, "ADI feature\nB×Cₗ×Hₗ×Wₗ", face="#ffedd9", edge=COLORS["geometry"], fontsize=8.5, weight="bold"
    )
    feature_stack(ax, 7.5, 23.0, COLORS["geometry"], scale=0.85)

    rounded_box(ax, 27, 46, 22, 9, "Shared 1×1 reduction\n3×3 Conv–BN–ReLU", face="#edf6fc", edge=COLORS["rgb"], fontsize=8.0)
    rounded_box(ax, 27, 12, 22, 9, "Shared 1×1 reduction\n3×3 Conv–BN–ReLU", face="#fff5ea", edge=COLORS["geometry"], fontsize=8.0)
    arrow(ax, (20, 50.5), (27, 50.5), color=COLORS["rgb"])
    arrow(ax, (20, 16.5), (27, 16.5), color=COLORS["geometry"])

    rounded_box(ax, 56, 46, 15, 9, "ECA\nRGB enhanced", face="#dff1ff", edge=COLORS["rgb"], fontsize=8.2, weight="bold")
    rounded_box(ax, 56, 12, 15, 9, "Coordinate attn.\nADI enhanced", face="#ffedd9", edge=COLORS["geometry"], fontsize=8.0, weight="bold")
    arrow(ax, (49, 50.5), (56, 50.5), color=COLORS["rgb"])
    arrow(ax, (49, 16.5), (56, 16.5), color=COLORS["geometry"])

    rounded_box(ax, 80, 49, 18, 8, "GAP + Wq\nQₗ: B×1×d", face="#e8f5ff", edge=COLORS["rgb"], fontsize=8.1, weight="bold")
    rounded_box(ax, 80, 8, 18, 8, "Wk, Wv + flatten\nKₗ: B×Nₗ×d\nVₗ: B×d×Nₗ", face="#fff1df", edge=COLORS["geometry"], fontsize=7.5, weight="bold")
    arrow(ax, (71, 50.5), (80, 53.0), color=COLORS["rgb"])
    arrow(ax, (71, 16.5), (80, 12.0), color=COLORS["geometry"])

    rounded_box(
        ax,
        106,
        43,
        22,
        11,
        "αₗ = softmax(QₗKₗᵀ/√d)\nattention: B×1×Nₗ",
        face="#efe4fa",
        edge=COLORS["fusion"],
        fontsize=8.3,
        weight="bold",
    )
    arrow(ax, (98, 53.0), (106, 49.0), color=COLORS["fusion"])
    arrow(
        ax,
        (98, 12.0),
        (109, 43.0),
        color=COLORS["fusion"],
        connection="angle3,angleA=0,angleB=-90",
    )

    rounded_box(
        ax,
        135,
        43,
        20,
        11,
        "Vₗαₗᵀ → Wo → LN\nbroadcast to Hₗ×Wₗ",
        face="#efe4fa",
        edge=COLORS["fusion"],
        fontsize=7.9,
        weight="bold",
    )
    arrow(ax, (128, 48.5), (135, 48.5), color=COLORS["fusion"])
    arrow(
        ax,
        (98, 12.0),
        (140, 43.0),
        color=COLORS["geometry"],
        connection="angle3,angleA=0,angleB=-90",
        width=1.0,
    )

    rounded_box(
        ax,
        80,
        24,
        25,
        10,
        "Concat(RGB, ADI)\n1×1 Conv–BN–sigmoid → gate gₗ",
        face="#f4f7fa",
        edge=COLORS["neutral"],
        fontsize=8.0,
        weight="bold",
    )
    arrow(ax, (71, 49.0), (84, 34.0), color=COLORS["rgb"], connection="angle3,angleA=0,angleB=90")
    arrow(ax, (71, 18.0), (84, 24.0), color=COLORS["geometry"], connection="angle3,angleA=0,angleB=-90")

    rounded_box(
        ax,
        112,
        24,
        25,
        10,
        "Mₗ = gₗ⊙RGB + (1−gₗ)⊙ADI\nadaptive modality blend",
        face="#eef5f7",
        edge=COLORS["neutral"],
        fontsize=7.8,
        weight="bold",
    )
    arrow(ax, (105, 29.0), (112, 29.0), color=COLORS["neutral"])

    # Cross feature includes the enhanced RGB residual exactly as implemented.
    rounded_box(
        ax,
        158,
        43,
        14,
        11,
        "Fcross\nRGB + context",
        face="#e9e2f5",
        edge=COLORS["fusion"],
        fontsize=7.9,
        weight="bold",
    )
    arrow(ax, (155, 48.5), (158, 48.5), color=COLORS["fusion"])
    arrow(
        ax,
        (71, 52.5),
        (164, 54.0),
        color=COLORS["rgb"],
        connection="angle3,angleA=0,angleB=90",
        width=1.0,
    )

    ax.add_patch(
        plt.Circle((145, 29), 2.4, facecolor="white", edgecolor=COLORS["fusion"], linewidth=1.6, zorder=3)
    )
    ax.text(145, 29, "+", ha="center", va="center", fontsize=14, fontweight="bold", color=COLORS["fusion"], zorder=4)
    arrow(ax, (137, 29.0), (142.5, 29.0), color=COLORS["neutral"])
    arrow(ax, (165, 43.0), (146.5, 31.0), color=COLORS["fusion"], connection="angle3,angleA=-90,angleB=0")

    rounded_box(
        ax,
        152,
        23,
        20,
        12,
        "3×3 Conv–BN–ReLU\nrestore Cₗ channels\nFused feature",
        face="#ddf5e9",
        edge=COLORS["decoder"],
        fontsize=8.0,
        weight="bold",
    )
    arrow(ax, (147.5, 29.0), (152, 29.0), color=COLORS["decoder"], width=1.4)

    ax.text(
        3,
        2.5,
        "Only B·Nₗ attention scores are materialized: score memory O(BNₗ), aggregation O(BNₗd).",
        fontsize=9.2,
        color=COLORS["fusion"],
        fontweight="bold",
        va="bottom",
    )

    png = figure_dir / "fig_msfm1.png"
    pdf = figure_dir / "fig_msfm.pdf"
    save_figure(fig, png, pdf, dpi=300)
    return [png, pdf]


def fix_robot_figure(figure_dir: Path, source_figure_dir: Path) -> tuple[Path, dict]:
    """Create a corrected reference copy without touching the manuscript image.

    The source composite has fixed dimensions (10206×2099). The G1 inset is
    x=[6832,7704), with the content beneath ``Depth`` at y=[619,1040) and
    the content beneath ``Segmentation`` at y=[1139,1559). The black-pixel
    ratio makes this operation idempotent: a depth visualization contains
    more invalid/black pixels than the green RGB overlay.
    """
    source_path = source_figure_dir / "real_experiment_all1.png"
    path = figure_dir / "real_experiment_all1.png"
    if source_path.resolve() == path.resolve():
        raise ValueError("Refusing to overwrite the manuscript robot figure")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    shutil.copy2(source_path, path)
    image = Image.open(path).convert("RGBA")
    if image.size != (10206, 2099):
        raise ValueError(f"Unexpected robot composite size {image.size}; expected (10206, 2099)")

    x0, x1 = 6832, 7704
    depth_box = (x0, 619, x1, 1040)
    seg_box = (x0, 1139, x1, 1559)
    depth_labeled = image.crop(depth_box)
    seg_labeled = image.crop(seg_box)

    def black_fraction(panel: Image.Image) -> float:
        arr = np.asarray(panel.convert("RGB"))
        return float(np.mean(np.all(arr < 18, axis=2)))

    before = {
        "depth_labeled_black_fraction": black_fraction(depth_labeled),
        "segmentation_labeled_black_fraction": black_fraction(seg_labeled),
    }
    swapped = before["depth_labeled_black_fraction"] < before["segmentation_labeled_black_fraction"]
    if swapped:
        image.paste(seg_labeled.resize(depth_labeled.size, Image.Resampling.BICUBIC), depth_box)
        image.paste(depth_labeled.resize(seg_labeled.size, Image.Resampling.BICUBIC), seg_box)
        image.save(path)

    checked = Image.open(path).convert("RGBA")
    after_depth = checked.crop(depth_box)
    after_seg = checked.crop(seg_box)
    after = {
        "depth_labeled_black_fraction": black_fraction(after_depth),
        "segmentation_labeled_black_fraction": black_fraction(after_seg),
    }
    if after["depth_labeled_black_fraction"] <= after["segmentation_labeled_black_fraction"]:
        raise RuntimeError("Unitree-G1 content validation failed after attempted swap")

    return path, {
        "operation": "swapped G1 Depth/Segmentation contents" if swapped else "already corrected",
        "g1_inset_x": [x0, x1],
        "depth_content_y": [619, 1040],
        "segmentation_content_y": [1139, 1559],
        "before": before,
        "after": after,
    }


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {}


def raw_road_mask(path: Path) -> np.ndarray:
    label = np.asarray(Image.open(path))
    if label.ndim == 3:
        return ((label[:, :, 2] > 200) & (label[:, :, 0] > 200)).astype(bool)
    return (label > 0).astype(bool)


def blend_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    result = rgb.astype(np.float32).copy()
    paint = np.asarray(color, dtype=np.float32)
    result[mask] = (1.0 - alpha) * result[mask] + alpha * paint
    return np.clip(result, 0, 255).astype(np.uint8)


def error_map(rgb: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    base = np.clip(rgb.astype(np.float32) * 0.34, 0, 255).astype(np.uint8)
    tp = prediction & target
    fp = prediction & ~target
    fn = ~prediction & target
    base[tp] = (38, 205, 74)
    base[fp] = (238, 52, 52)
    base[fn] = (44, 120, 255)
    return base


def per_image_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum(prediction & target))
    fp = int(np.sum(prediction & ~target))
    fn = int(np.sum(~prediction & target))
    denom_f = 2 * tp + fp + fn
    denom_iou = tp + fp + fn
    f1 = (2.0 * tp / denom_f) if denom_f else 1.0
    iou = (tp / denom_iou) if denom_iou else 1.0
    return {"F1": f1, "IoU": iou, "TP": tp, "FP": fp, "FN": fn}


def generate_qualitative(
    figure_dir: Path,
    *,
    data_root: Path,
    split_file: Path,
    checkpoint_path: Path,
    sample_ids: Iterable[str],
    threshold: float,
    device: str,
) -> tuple[list[Path], dict]:
    sample_ids = tuple(sample_ids)
    if len(sample_ids) != 4:
        raise ValueError("Exactly four sample IDs are required for the 4×4 qualitative figure")

    split_ids = {
        line.strip().removesuffix(".png")
        for line in split_file.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(set(sample_ids) - split_ids)
    if missing:
        raise ValueError(f"Qualitative sample IDs are not in the fixed validation manifest: {missing}")
    categories = {sid.split("_", 1)[0] for sid in sample_ids}
    if not {"um", "umm", "uu"}.issubset(categories):
        raise ValueError(f"Qualitative selection must cover UM, UMM, and UU; got {sorted(categories)}")

    dataset = KITTIRoadDataset(
        str(data_root),
        split="val",
        category="all",
        img_h=384,
        img_w=1248,
        use_augmentation=False,
        split_file=str(split_file),
    )
    index_by_name = {sample["name"]: index for index, sample in enumerate(dataset.samples)}

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for figure inference but is unavailable")
    model = get_ablation_model("full", pretrained=False)
    checkpoint = load_checkpoint(model, checkpoint_path)
    model.to(requested_device).eval()

    fig, axes = plt.subplots(4, 4, figsize=(16.0, 7.9), facecolor="white")
    titles = ["(a) RGB image", "(b) Stored ADI", "(c) LiteViLNet prediction", "(d) Error map: TP / FP / FN"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=11.2, fontweight="bold", pad=7)

    records = []
    panel_paths: list[Path] = []
    with torch.inference_mode():
        for row, sample_id in enumerate(sample_ids):
            dataset_item = dataset[index_by_name[sample_id]]
            sample = dataset.samples[index_by_name[sample_id]]
            rgb = np.asarray(Image.open(sample["rgb"]).convert("RGB"))
            adi = np.asarray(Image.open(sample["adi"]).convert("RGB"))
            target = raw_road_mask(Path(sample["label"]))

            rgb_tensor = dataset_item["rgb"].unsqueeze(0).to(requested_device)
            adi_tensor = dataset_item["adi"].unsqueeze(0).to(requested_device)
            logits = model(rgb_tensor, adi_tensor)
            logits = F.interpolate(
                logits,
                size=target.shape,
                mode="bilinear",
                align_corners=False,
            )
            probability = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
            prediction = probability >= threshold
            metrics = per_image_metrics(prediction, target)

            overlay = blend_mask(rgb, prediction, (32, 210, 92), alpha=0.52)
            errors = error_map(rgb, prediction, target)
            panels = [rgb, adi, overlay, errors]

            sample_asset_dir = figure_dir / "qualitative_panels" / sample_id
            sample_asset_dir.mkdir(parents=True, exist_ok=True)
            assets = {
                "rgb": rgb,
                "stored_adi": adi,
                "prediction_overlay": overlay,
                "binary_prediction": prediction.astype(np.uint8) * 255,
                "probability": np.clip(probability * 255.0, 0, 255).astype(np.uint8),
                "error_map": errors,
            }
            sample_paths: dict[str, str] = {}
            for asset_name, asset in assets.items():
                asset_path = sample_asset_dir / f"{asset_name}.png"
                Image.fromarray(asset).save(asset_path)
                panel_paths.append(asset_path)
                sample_paths[asset_name] = str(asset_path.resolve())

            for col, panel in enumerate(panels):
                axes[row, col].imshow(panel, interpolation="nearest")
                axes[row, col].axis("off")

            axes[row, 0].text(
                0.012,
                0.965,
                sample_id,
                transform=axes[row, 0].transAxes,
                ha="left",
                va="top",
                fontsize=8.4,
                fontweight="bold",
                color="white",
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "black", "alpha": 0.68, "edgecolor": "none"},
            )
            axes[row, 3].text(
                0.985,
                0.965,
                f"F1={metrics['F1']:.3f}\nIoU={metrics['IoU']:.3f}",
                transform=axes[row, 3].transAxes,
                ha="right",
                va="top",
                fontsize=8.4,
                fontweight="bold",
                color="white",
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "black", "alpha": 0.72, "edgecolor": "none"},
            )

            records.append(
                {
                    "sample_id": sample_id,
                    "category": sample_id.split("_", 1)[0].upper(),
                    "rgb": str(Path(sample["rgb"]).resolve()),
                    "rgb_sha256": sha256(Path(sample["rgb"])),
                    "adi": str(Path(sample["adi"]).resolve()),
                    "adi_sha256": sha256(Path(sample["adi"])),
                    "label": str(Path(sample["label"]).resolve()),
                    "label_sha256": sha256(Path(sample["label"])),
                    "manual_drawing_assets": sample_paths,
                    "metrics_original_resolution": metrics,
                }
            )

    plt.subplots_adjust(left=0.006, right=0.994, top=0.935, bottom=0.008, wspace=0.012, hspace=0.045)
    png = figure_dir / "fig_qualitative.png"
    pdf = figure_dir / "fig_qualitative.pdf"
    save_figure(fig, png, pdf, dpi=260)

    return [png, pdf, *panel_paths], {
        "dataset": "KITTI Road training images with fixed local validation manifest",
        "split_file": str(split_file.resolve()),
        "split_file_sha256": sha256(split_file),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_metric": checkpoint.get("best_metric"),
        "config": "full",
        "seed": 42,
        "network_input_resolution": [384, 1248],
        "rendered_prediction_resolution": "original sample resolution via bilinear logit resize",
        "threshold": threshold,
        "threshold_source": "seed-42 validation-global BestThreshold",
        "samples": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("all", "architecture", "msfm", "robot", "qualitative"),
        default="all",
        help="Generate one asset group or all revision figures.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Reference-material output directory; must differ from the manuscript figure directory.",
    )
    parser.add_argument(
        "--source-figure-dir",
        type=Path,
        default=DEFAULT_SOURCE_FIGURE_DIR,
        help="Read-only source directory for the author's existing manuscript figures.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sample-ids", nargs=4, default=DEFAULT_SAMPLE_IDS)
    parser.add_argument("--threshold", type=float, default=0.66)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.figure_dir.resolve() == args.source_figure_dir.resolve():
        raise ValueError("Refusing to write generated material into the manuscript figure directory")
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    details: dict[str, object] = {}

    if args.only in ("all", "architecture"):
        paths = generate_architecture(args.figure_dir)
        generated.extend(paths)
        details["architecture"] = {
            "contract": "MobileNetV3 RGB stages; Conv stem + four DSConv geometry stages; five MSFMs; LKB; U-Net decoder",
            "outputs": [str(path.resolve()) for path in paths],
        }
    if args.only in ("all", "msfm"):
        paths = generate_msfm(args.figure_dir)
        generated.extend(paths)
        details["msfm"] = {
            "contract": "one pooled RGB query, spatial ADI K/V, B×1×N_l attention, gate blend + cross residual + output convolution",
            "outputs": [str(path.resolve()) for path in paths],
        }
    if args.only in ("all", "robot"):
        path, robot_details = fix_robot_figure(args.figure_dir, args.source_figure_dir)
        generated.append(path)
        details["robot"] = robot_details
    if args.only in ("all", "qualitative"):
        paths, qualitative_details = generate_qualitative(
            args.figure_dir,
            data_root=args.data_root,
            split_file=args.split_file,
            checkpoint_path=args.checkpoint,
            sample_ids=args.sample_ids,
            threshold=args.threshold,
            device=args.device,
        )
        generated.extend(paths)
        details["qualitative"] = qualitative_details

    unique_generated = list(dict.fromkeys(path.resolve() for path in generated))
    payload = {
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__)),
        "requested_group": args.only,
        "details": details,
        "outputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in unique_generated
        ],
    }
    save_json(args.manifest, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
