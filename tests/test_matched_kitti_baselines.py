import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from tools.cache_official_sne_normals import compute_one as compute_kitti_normal
from tools.cache_official_orfd_normals import (
    collect_tasks as collect_orfd_normal_tasks,
    compute_one as compute_orfd_normal,
    validate_formal_calibration_metadata,
)
from tools.prepare_matched_kitti_baselines import read_manifest, source_path
from tools.sanitize_table1_supplement import anonymize_string, sanitize_results, scan_tree
from tools.train_matched_kitti_baseline import MatchedKITTINormalDataset, load_sne_roadseg
from tools.train_matched_kitti_offnet import MatchedKITTIOFFNetDataset
from tools.train_matched_orfd_baseline import (
    ORFDNormalDataset,
    fixed_metrics,
    forward_probability,
)
from tools.train_matched_orfd_roadformer import evaluate as evaluate_orfd_roadformer


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
        ("/workspace/third_party/matched_baselines/OFF-Net", "${OFFNET_SOURCE}"),
    ],
)
def test_new_official_source_paths_are_anonymized(source: str, expected: str) -> None:
    assert anonymize_string(source) == expected


def test_orfd_adapter_preserves_released_label_and_offnet_scaling(tmp_path: Path) -> None:
    data_root = tmp_path / "orfd"
    normal_root = tmp_path / "normals"
    for split in ("training", "validation", "testing"):
        for directory in ("image_data", "gt_image"):
            (data_root / split / directory).mkdir(parents=True)
        (normal_root / split / "normal").mkdir(parents=True)
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        Image.fromarray(rgb).save(data_root / split / "image_data" / "100.png")
        label = np.zeros((4, 4, 3), dtype=np.uint8)
        label[0, 1] = 255
        Image.fromarray(label).save(data_root / split / "gt_image" / "100_fillcolor.png")
        np.save(
            normal_root / split / "normal" / "100.npy",
            np.zeros((3, 4, 4), dtype=np.float32),
            allow_pickle=False,
        )

    dataset = ORFDNormalDataset(data_root, normal_root, "test", "offnet", 4, 4)
    sample = dataset[0]
    assert sample["label"][0].tolist() == [0, 1, 0, 0]
    assert sample["label_original"].tolist() == sample["label"].tolist()
    assert sample["label_training"].shape == (1, 1)
    assert torch.allclose(sample["rgb"][0], torch.ones((4, 4)))
    assert tuple(sample["normal"].shape) == (3, 4, 4)


def test_kitti_offnet_adapter_uses_separate_offnet_normal_cache(tmp_path: Path) -> None:
    data_root = tmp_path / "matched"
    normal_root = tmp_path / "offnet_normals"
    for directory in ("image_2", "gt_image_2"):
        (data_root / "training" / directory).mkdir(parents=True)
    (normal_root / "training" / "normal").mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
        data_root / "training" / "image_2" / "um_000001.png"
    )
    label = np.zeros((4, 4, 3), dtype=np.uint8)
    label[0, 0] = (255, 0, 255)
    Image.fromarray(label).save(
        data_root / "training" / "gt_image_2" / "um_road_000001.png"
    )
    normal = np.ones((3, 4, 4), dtype=np.float32)
    np.save(normal_root / "training" / "normal" / "um_000001.npy", normal)
    dataset = MatchedKITTIOFFNetDataset(data_root, normal_root, "training", 4, 4)
    sample = dataset[0]
    assert sample["label"][0].tolist() == [1, 0, 0, 0]
    assert sample["label_training"].shape == (1, 1)
    assert torch.allclose(sample["normal"], torch.ones((3, 4, 4)))


def test_fixed_orfd_metrics_are_accumulated_from_counts() -> None:
    metrics = fixed_metrics({"tp": 8, "fp": 2, "tn": 9, "fn": 1})
    assert metrics["PRE"] == pytest.approx(0.8)
    assert metrics["REC"] == pytest.approx(8 / 9)
    assert metrics["F_score"] == pytest.approx(16 / 19)
    assert metrics["IoU"] == pytest.approx(8 / 11)


def test_offnet_four_microbatches_match_one_global_mean_loss_update() -> None:
    """Physical batch 2 x four accumulation matches global batch 8 gradients."""
    generator = torch.Generator().manual_seed(20260805)
    inputs = torch.randn((8, 3, 3, 5), generator=generator)
    targets = torch.randint(0, 2, (8, 3, 5), generator=generator)
    global_model = torch.nn.Conv2d(3, 2, kernel_size=1, bias=False)
    micro_model = torch.nn.Conv2d(3, 2, kernel_size=1, bias=False)
    micro_model.load_state_dict(global_model.state_dict())
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=0.01)
    micro_optimizer = torch.optim.SGD(micro_model.parameters(), lr=0.01)

    global_optimizer.zero_grad(set_to_none=True)
    global_loss = torch.nn.functional.cross_entropy(global_model(inputs), targets)
    global_loss.backward()
    global_gradient = global_model.weight.grad.detach().clone()
    global_optimizer.step()

    micro_optimizer.zero_grad(set_to_none=True)
    for start in range(0, 8, 2):
        micro_loss = torch.nn.functional.cross_entropy(
            micro_model(inputs[start : start + 2]), targets[start : start + 2]
        )
        (micro_loss / 4).backward()
    micro_gradient = micro_model.weight.grad.detach().clone()
    micro_optimizer.step()

    assert torch.allclose(micro_gradient, global_gradient, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        micro_model.weight, global_model.weight, atol=1e-7, rtol=1e-6
    )


def test_orfd_usnet_probability_restores_spatial_shape() -> None:
    class FlatUSNet(torch.nn.Module):
        def forward(self, rgb: torch.Tensor, _normal: torch.Tensor):
            pixels = rgb.shape[0] * rgb.shape[2] * rgb.shape[3]
            alpha = torch.tensor([[1.0, 3.0]]).repeat(pixels, 1)
            return (None, None, None, None, None, alpha)

    rgb = torch.zeros((2, 3, 4, 5))
    normal = torch.zeros_like(rgb)
    probability, _ = forward_probability("usnet", FlatUSNet(), rgb, normal)
    assert probability.shape == (2, 4, 5)
    assert torch.allclose(probability, torch.full((2, 4, 5), 0.75))


def test_orfd_roadformer_evaluates_native_logits_before_gt_restoration() -> None:
    class FakeRoadFormer(torch.nn.Module):
        def data_preprocessor(self, batch, _training):
            return {
                "inputs": torch.zeros((1, 6, 2, 3)),
                "data_samples": batch["data_samples"],
            }

        def encode_decode(self, inputs, batch_img_metas):
            assert inputs.shape == (1, 6, 2, 3)
            assert batch_img_metas == [{"ori_shape": (4, 3)}]
            foreground = torch.tensor([[[4.0, -4.0, 4.0], [-4.0, 4.0, -4.0]]])
            return torch.stack((-foreground, foreground), dim=1)

    target = torch.tensor(
        [[[1, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0]]], dtype=torch.long
    )
    sample = SimpleNamespace(
        metainfo={"ori_shape": (4, 3)},
        gt_sem_seg=SimpleNamespace(data=target),
    )
    result = evaluate_orfd_roadformer(
        FakeRoadFormer(), [{"data_samples": [sample]}]
    )
    assert result["threshold_swept_grid"] == [2, 3]
    assert result["fixed_argmax_ground_truth_grid"] == [4, 3]
    fixed = result["official_fixed_argmax"]
    assert sum(fixed[key] for key in ("tp", "fp", "tn", "fn")) == 12
    assert fixed["F_score"] == pytest.approx(1.0)


def test_orfd_normal_cache_rejects_invalid_existing_array(tmp_path: Path) -> None:
    output = tmp_path / "normal.npy"
    np.save(output, np.zeros((3, 4, 5), dtype=np.float32), allow_pickle=False)
    task = (
        str(tmp_path / "unused_depth.png"),
        str(output),
        np.eye(3, dtype=np.float32).tolist(),
        False,
        False,
        1000.0,
    )
    with pytest.raises(ValueError, match="Invalid existing normal cache"):
        compute_orfd_normal(task)


def test_orfd_formal_cache_requires_timestamp_matched_calibration(tmp_path: Path) -> None:
    data_root = tmp_path / "orfd"
    for split, timestamp in (("training", 100), ("validation", 101), ("testing", 102)):
        (data_root / split / "dense_depth").mkdir(parents=True)
        (data_root / split / "calib").mkdir(parents=True)
        Image.fromarray(np.ones((2, 3), dtype=np.uint16)).save(
            data_root / split / "dense_depth" / f"{timestamp}.png"
        )
    (data_root / "training" / "calib" / "100.txt").write_text(
        "cam_K: 1 0 0 0 1 0 0 0 1\n", encoding="utf-8"
    )
    args = SimpleNamespace(
        data_root=data_root,
        output_root=tmp_path / "normals",
        profile="sne_roadseg",
        include_flipped_training=True,
        force=False,
        require_exact_calibration=True,
    )
    with pytest.raises(ValueError, match="2 of 3 samples"):
        collect_orfd_normal_tasks(args)


def test_orfd_formal_metadata_requires_complete_exact_calibration() -> None:
    valid = {
        "calibration": {
            "released_calibration_files": 11830,
            "exact_sample_matches": 11830,
            "nearest_timestamp_matches": 0,
            "maximum_nearest_timestamp_gap": 0,
            "inferred_intrinsic_counts": {},
        }
    }
    validate_formal_calibration_metadata(valid)
    invalid = json.loads(json.dumps(valid))
    invalid["calibration"]["exact_sample_matches"] = 11829
    invalid["calibration"]["nearest_timestamp_matches"] = 1
    with pytest.raises(ValueError, match="complete released per-frame calibration"):
        validate_formal_calibration_metadata(invalid)


def test_kitti_normal_cache_rejects_invalid_existing_array(tmp_path: Path) -> None:
    output = tmp_path / "normal.npy"
    np.save(output, np.zeros((3, 4, 5), dtype=np.float32), allow_pickle=False)
    task = (
        str(tmp_path / "unused_depth.png"),
        str(tmp_path / "unused_calib.txt"),
        str(output),
        False,
        False,
    )
    with pytest.raises(ValueError, match="Invalid existing normal cache"):
        compute_kitti_normal(task)
