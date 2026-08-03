import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from tools.prepare_matched_kitti_baselines import read_manifest, source_path
from tools.sanitize_table1_supplement import anonymize_string, sanitize_results, scan_tree
from tools.train_matched_kitti_baseline import MatchedKITTINormalDataset, load_sne_roadseg


def test_manifest_rejects_duplicates(tmp_path: Path) -> None:
    manifest = tmp_path / "split.txt"
    manifest.write_text("um_000001\num_000001.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(manifest)


def test_source_path_uses_kitti_label_naming(tmp_path: Path) -> None:
    label = source_path(tmp_path / "kitti", tmp_path / "depth", "gt_image_2", "umm_000123")
    depth = source_path(tmp_path / "kitti", tmp_path / "depth", "depth_u16", "umm_000123")
    assert label == tmp_path / "kitti" / "training" / "gt_image_2" / "umm_road_000123.png"
    assert depth == tmp_path / "depth" / "training" / "umm_000123.png"


def test_matched_dataset_uses_litevilnet_road_mask(tmp_path: Path) -> None:
    split = tmp_path / "validation"
    for subdir in ("image_2", "gt_image_2", "normal"):
        (split / subdir).mkdir(parents=True)

    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(split / "image_2" / "um_000001.png")
    label = np.zeros((2, 3, 3), dtype=np.uint8)
    label[0, 0] = (255, 0, 255)  # KITTI magenta road: positive.
    label[0, 1] = (255, 0, 0)  # Red-only valid non-road: negative.
    Image.fromarray(label).save(split / "gt_image_2" / "um_road_000001.png")
    np.save(split / "normal" / "um_000001.npy", np.zeros((3, 2, 3), dtype=np.float32))

    dataset = MatchedKITTINormalDataset(tmp_path, "validation", "sne_roadseg", height=2, width=3)
    sample = dataset[0]
    assert tuple(sample["rgb"].shape) == (3, 2, 3)
    assert tuple(sample["normal"].shape) == (3, 2, 3)
    assert sample["label"].tolist() == [[1, 0, 0], [0, 0, 0]]


def test_sne_loader_returns_complete_model_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models_module = ModuleType("models")
    networks_module = ModuleType("models.networks")

    def define_roadseg(*_args: object, **_kwargs: object) -> torch.nn.Module:
        return torch.nn.Conv2d(6, 2, kernel_size=1)

    networks_module.define_RoadSeg = define_roadseg  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "models", models_module)
    monkeypatch.setitem(sys.modules, "models.networks", networks_module)
    bundle = load_sne_roadseg(
        SimpleNamespace(official_source=tmp_path, learning_rate=None),
        torch.device("cpu"),
    )
    assert isinstance(bundle.model, torch.nn.Module)
    assert isinstance(bundle.optimizer, torch.optim.SGD)
    assert bundle.scheduler is not None
    assert {path.name for path in bundle.official_files} >= {"networks.py", "roadseg_model.py"}


def test_supplement_result_copy_is_anonymous(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = {
        "best_checkpoint": "/" + "home/researcher/work/LiteViLNet/runs/formal/best_model.pth",
        "official_source": "/" + "data/Database/project/other_models/USNet/source",
        "system": {"hostname": "ai" + "hub", "device": "RTX 4090 D"},
        "metrics": {"MaxF": 0.97},
        "checkpoint_sha256": "abc123",
    }
    (source / "seed.json").write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "anonymous"
    sanitize_results(source, output)
    copied = json.loads((output / "seed.json").read_text(encoding="utf-8"))
    assert copied["best_checkpoint"] == "runs/formal/best_model.pth"
    assert copied["official_source"] == "${USNET_SOURCE}"
    assert copied["metrics"] == payload["metrics"]
    assert copied["checkpoint_sha256"] == payload["checkpoint_sha256"]
    assert copied["system"] == {"hostname": "anonymous-host", "device": "RTX 4090 D"}
    assert copied["supplement_anonymization"]["metrics_and_hashes_changed"] is False
    scan_tree(output)


def test_supplement_scan_rejects_identity_leak(tmp_path: Path) -> None:
    synthetic_email = "person" + "@" + "example.edu"
    (tmp_path / "leak.txt").write_text(f"contact: {synthetic_email}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="email address"):
        scan_tree(tmp_path)


def test_supplement_scan_honors_private_deny_token(tmp_path: Path) -> None:
    (tmp_path / "metadata.txt").write_text("private-lab-token", encoding="utf-8")
    with pytest.raises(RuntimeError, match="supplied deny token"):
        scan_tree(tmp_path, ("private-lab-token",))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("/workspace/third_party/matched_baselines/PLARD", "${PLARD_SOURCE}"),
        ("/workspace/third_party/matched_baselines/Road-Former", "${ROADFORMER_SOURCE}"),
    ],
)
def test_new_official_source_paths_are_anonymized(source: str, expected: str) -> None:
    assert anonymize_string(source) == expected
