#!/usr/bin/env python3
"""Validate and summarize multi-seed local ORFD baseline reproductions."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sanitize_table1_supplement import sanitize
from tools.cache_official_orfd_normals import validate_formal_calibration_metadata
from tools.summarize_matched_kitti_baselines import (
    PROVENANCE_PATH,
    accepted_source_hashes,
    file_sha256,
)


DEFAULT_METHODS = ("usnet", "sne_roadseg", "offnet", "roadformer")
COMMITS = {
    "usnet": "d761158ad42df7dcb62fa257dd02ce11c85f94a5",
    "sne_roadseg": "5e7900bfd59887634ced687ffe85a73018a38659",
    "offnet": "50e63d24836198e8fb5af707e521f414104b4876",
    "roadformer": "f675a3467cb168ebc727648390c304279bbcb079",
}
REMOTES = {
    "usnet": "https://github.com/morancyc/USNet.git",
    "sne_roadseg": "https://github.com/hlwang1124/SNE-RoadSeg.git",
    "offnet": "https://github.com/chaytonmin/Off-Road-Freespace-Detection",
    "roadformer": "https://github.com/LiJiahang617/Road-Former.git",
}
METHOD_PROTOCOL = {
    "usnet": {
        "epochs_requested": 30,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 2,
        "amp": True,
    },
    "sne_roadseg": {
        "epochs_requested": 30,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 2,
        "amp": True,
    },
    "offnet": {
        "epochs_requested": 30,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 8,
        "amp": False,
    },
    "roadformer": {
        "epochs_requested": 50,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 4,
        "amp": False,
    },
}
PARAMETERS = {
    "usnet": 30_738_444,
    "sne_roadseg": 201_324_806,
    "offnet": 25_209_608,
    "roadformer": 206_860_175,
}
REQUIRED_SOURCE_FILES = {
    "usnet": {"model/usnet.py", "loss.py"},
    "sne_roadseg": {"models/networks.py", "models/roadseg_model.py"},
    "offnet": {
        "models/transformer_models/backbones/transformer.py",
        "models/transformer_models/decode_heads/head.py",
        "models/loss.py",
        "options/base_options.py",
        "options/train_options.py",
        "scripts/train.sh",
    },
    "roadformer": {
        "configs/roadformer_orfd/roadformer_convnext-b_orfd-352x640.py",
        "mmcv_custom/transforms/loading.py",
        "mmseg_custom/datasets/mmorfd.py",
        "mmpretrain_custom/models/backbones/twin_convnext.py",
    },
}
ORIGINAL_TEST_PIXELS = 2193 * 720 * 1280


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-seeds", default="40,41,42")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed-output-dir", type=Path)
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated official baselines to summarize (default: all four).",
    )
    return parser.parse_args()


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.expected_seeds.split(",")]
    if seeds != sorted(set(seeds)):
        raise ValueError(f"Expected seeds must be unique and sorted: {seeds}")
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    if not methods or any(method not in DEFAULT_METHODS for method in methods):
        raise ValueError(f"Unknown ORFD method in --methods: {methods}")
    if len(set(methods)) != len(methods):
        raise ValueError(f"Duplicate ORFD method in --methods: {methods}")
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "protocol": "local ORFD released-split retraining",
        "seeds": seeds,
        "methods": {},
    }
    rows: list[dict[str, Any]] = []
    for method in methods:
        results = []
        for seed in seeds:
            path = args.input_root / f"{method}_seed{seed}" / "result.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            result = json.loads(path.read_text(encoding="utf-8"))
            checks = {
                "protocol": (result["protocol"], "local ORFD released-split retraining"),
                "baseline": (result["baseline"], method),
                "seed": (int(result["seed"]), seed),
                "input_size": (result["input_size"], [704, 1280]),
                "train_count": (int(result["train_count"]), 8392),
                "val_count": (int(result["val_count"]), 1245),
                "test_count": (int(result["test_count"]), 2193),
                "epochs_requested": (
                    int(result["epochs_requested"]),
                    METHOD_PROTOCOL[method]["epochs_requested"],
                ),
                "batch_size": (
                    int(result["batch_size"]), METHOD_PROTOCOL[method]["batch_size"]
                ),
                "gradient_accumulation_steps": (
                    int(result.get("gradient_accumulation_steps", 1)),
                    METHOD_PROTOCOL[method]["gradient_accumulation_steps"],
                ),
                "effective_batch_size": (
                    int(
                        result.get(
                            "effective_batch_size",
                            int(result["batch_size"])
                            * int(result.get("gradient_accumulation_steps", 1)),
                        )
                    ),
                    METHOD_PROTOCOL[method]["effective_batch_size"],
                ),
                "amp": (bool(result["amp"]), METHOD_PROTOCOL[method]["amp"]),
                "checkpoint_selection": (
                    result["checkpoint_selection"],
                    "validation 101-threshold MaxF",
                ),
                "official_remote": (result["official_remote"], REMOTES[method]),
                "official_commit": (result["official_commit"], COMMITS[method]),
                "official_semantic_diff_clean": (
                    bool(result["official_semantic_diff_clean"]),
                    True,
                ),
                "testing_samples": (int(result["testing"]["samples"]), 2193),
                "parameters": (int(result["parameters"]), PARAMETERS[method]),
            }
            mismatches = {
                key: pair for key, pair in checks.items() if pair[0] != pair[1]
            }
            if method == "offnet":
                offnet_step_checks = {
                    "physical_training_steps": (
                        int(result.get("physical_training_steps", -1)), 125_880
                    ),
                    "optimizer_steps": (int(result.get("optimizer_steps", -1)), 31_470),
                }
                mismatches.update(
                    {
                        key: pair
                        for key, pair in offnet_step_checks.items()
                        if pair[0] != pair[1]
                    }
                )
            if mismatches:
                raise ValueError(f"Invalid {method} seed {seed}: {mismatches}")
            best_epoch = int(result["best_epoch"])
            if not 1 <= best_epoch <= int(result["epochs_requested"]):
                raise ValueError(f"Invalid {method} best epoch for seed {seed}: {best_epoch}")

            checkpoint_hash = result.get("best_checkpoint_sha256")
            if not isinstance(checkpoint_hash, str) or not re.fullmatch(
                r"[0-9a-f]{64}", checkpoint_hash
            ):
                raise ValueError(f"Invalid {method} checkpoint hash for seed {seed}")
            checkpoint_path = Path(result["best_checkpoint"])
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            if file_sha256(checkpoint_path) != checkpoint_hash:
                raise ValueError(f"Checkpoint hash mismatch for {method} seed {seed}")

            expected_hashes = accepted_source_hashes(provenance, method)
            recorded_hashes = result["official_file_sha256"]
            recorded_files = set(recorded_hashes)
            if not REQUIRED_SOURCE_FILES[method].issubset(recorded_files):
                raise ValueError(f"{method} result omits required official source files")
            if not recorded_files.issubset(expected_hashes):
                raise ValueError(f"{method} result contains an unrecognized official source file")
            for relative_path, recorded_hash in recorded_hashes.items():
                if recorded_hash not in expected_hashes[relative_path]:
                    raise ValueError(
                        f"Unrecognized official source hash for {method}: {relative_path}"
                    )

            fixed = result["testing"]["official_fixed_argmax"]
            swept_grid = result["testing"].get("threshold_swept_grid")
            fixed_grid = result["testing"].get("fixed_argmax_ground_truth_grid")
            if swept_grid is not None and swept_grid != [704, 1280]:
                raise ValueError(
                    f"{method} seed {seed} threshold sweep uses {swept_grid}, "
                    "expected [704, 1280]"
                )
            if fixed_grid is not None and fixed_grid != [720, 1280]:
                raise ValueError(
                    f"{method} seed {seed} fixed evaluator uses {fixed_grid}, "
                    "expected [720, 1280]"
                )
            # OFF-Net and RoadFormer start only after the final grid metadata
            # was added. Earlier already-running USNet/SNE processes are
            # semantically identical but cannot acquire new result fields.
            if method in {"offnet", "roadformer"} and (
                swept_grid is None or fixed_grid is None
            ):
                raise ValueError(f"{method} seed {seed} omits evaluator-grid metadata")
            pixel_count = sum(int(fixed[key]) for key in ("tp", "fp", "tn", "fn"))
            if pixel_count != ORIGINAL_TEST_PIXELS:
                raise ValueError(
                    f"{method} seed {seed} fixed evaluator covers {pixel_count} pixels, "
                    f"expected {ORIGINAL_TEST_PIXELS}"
                )
            if method == "offnet":
                authors_fixed = result["testing"].get(
                    "authors_released_fixed_argmax"
                )
                if not isinstance(authors_fixed, dict):
                    raise ValueError(
                        f"OFF-Net seed {seed} omits the authors' literal test.py metric"
                    )
                authors_pixel_count = sum(
                    int(authors_fixed[key]) for key in ("tp", "fp", "tn", "fn")
                )
                if authors_pixel_count != ORIGINAL_TEST_PIXELS:
                    raise ValueError(
                        f"OFF-Net seed {seed} authors' evaluator covers "
                        f"{authors_pixel_count} pixels, expected {ORIGINAL_TEST_PIXELS}"
                    )
            if method == "roadformer":
                split_metadata = result.get("split_metadata", {})
                if split_metadata.get("counts") != {
                    "training": 8392,
                    "validation": 1245,
                    "testing": 2193,
                }:
                    raise ValueError("RoadFormer prepared split metadata is incomplete")
                if split_metadata.get("normal_source") != "official SNE-RoadSeg float32 cache":
                    raise ValueError("RoadFormer prepared inputs use an unexpected normal source")
                source_metadata = split_metadata.get("normal_source_metadata", {})
                validate_formal_calibration_metadata(source_metadata)
                accepted_sne_hashes = accepted_source_hashes(provenance, "sne_roadseg")[
                    "models/sne_model.py"
                ]
                roadformer_normal_checks = {
                    "profile": (source_metadata.get("profile"), "sne_roadseg"),
                    "official_commit": (
                        source_metadata.get("official_commit"), COMMITS["sne_roadseg"]
                    ),
                    "official_sne_sha256": (
                        source_metadata.get("official_sne_sha256") in accepted_sne_hashes,
                        True,
                    ),
                    "task_count": (int(source_metadata.get("task_count", -1)), 20222),
                    "depth_divisor": (int(source_metadata.get("depth_divisor", -1)), 1000),
                }
                roadformer_normal_mismatches = {
                    key: values
                    for key, values in roadformer_normal_checks.items()
                    if values[0] != values[1]
                }
                if roadformer_normal_mismatches:
                    raise ValueError(
                        "RoadFormer prepared normal metadata mismatch: "
                        f"{roadformer_normal_mismatches}"
                    )
            else:
                normal_metadata = result.get("normal_cache_metadata", {})
                validate_formal_calibration_metadata(normal_metadata)
                expected_profile = "offnet" if method == "offnet" else "sne_roadseg"
                expected_normal_commit = (
                    COMMITS["offnet"] if method == "offnet" else COMMITS["sne_roadseg"]
                )
                if normal_metadata.get("profile") != expected_profile:
                    raise ValueError(f"{method} uses the wrong normal-cache profile")
                if normal_metadata.get("official_commit") != expected_normal_commit:
                    raise ValueError(f"{method} normal cache uses an unexpected official commit")
                normal_method = "offnet" if method == "offnet" else "sne_roadseg"
                accepted_sne_hashes = accepted_source_hashes(provenance, normal_method)[
                    "models/sne_model.py"
                ]
                if normal_metadata.get("official_sne_sha256") not in accepted_sne_hashes:
                    raise ValueError(f"{method} normal cache uses an unexpected SNE file")
                expected_task_count = 11830 if method == "offnet" else 20222
                if int(normal_metadata.get("task_count", -1)) != expected_task_count:
                    raise ValueError(f"{method} normal cache is incomplete")
                expected_divisor = 256 if method == "offnet" else 1000
                if int(normal_metadata.get("depth_divisor", -1)) != expected_divisor:
                    raise ValueError(f"{method} normal cache uses an unexpected depth divisor")
                if method == "usnet" and normal_metadata.get(
                    "training_flip_semantics"
                ) != "depth is flipped before official SNE":
                    raise ValueError("USNet normal cache omits the official pre-SNE flip")
            results.append(result)
            if args.seed_output_dir:
                args.seed_output_dir.mkdir(parents=True, exist_ok=True)
                copy = sanitize(result)
                copy["supplement_anonymization"] = {
                    "local_absolute_paths": "replaced by portable placeholders",
                    "metrics_and_hashes_changed": False,
                }
                (args.seed_output_dir / f"{method}_orfd_seed{seed}.json").write_text(
                    json.dumps(copy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )

        fixed_keys = ("F_score", "PRE", "REC", "IoU", "GlobalAcc")
        swept_keys = ("MaxF", "AP", "PRE", "REC", "IoU")
        fixed = {
            key: mean_sd(
                [float(item["testing"]["official_fixed_argmax"][key]) for item in results]
            )
            for key in fixed_keys
        }
        swept = {
            key: mean_sd(
                [float(item["testing"]["threshold_swept"][key]) for item in results]
            )
            for key in swept_keys
        }
        method_summary = {
            "seeds": seeds,
            "epochs_requested": sorted({int(item["epochs_requested"]) for item in results}),
            "batch_size": sorted({int(item["batch_size"]) for item in results}),
            "gradient_accumulation_steps": sorted(
                {int(item.get("gradient_accumulation_steps", 1)) for item in results}
            ),
            "effective_batch_size": sorted(
                {
                    int(
                        item.get(
                            "effective_batch_size",
                            int(item["batch_size"])
                            * int(item.get("gradient_accumulation_steps", 1)),
                        )
                    )
                    for item in results
                }
            ),
            "parameters": int(results[0]["parameters"]),
            "official_commit": COMMITS[method],
            "testing_fixed_argmax": fixed,
            "testing_threshold_swept": swept,
            "best_epochs": [int(item["best_epoch"]) for item in results],
        }
        if method == "offnet":
            method_summary["authors_released_fixed_argmax"] = {
                key: mean_sd(
                    [
                        float(item["testing"]["authors_released_fixed_argmax"][key])
                        for item in results
                    ]
                )
                for key in fixed_keys
            }
        if len({int(item["parameters"]) for item in results}) != 1:
            raise ValueError(f"Parameter count differs across {method} seeds")
        summary["methods"][method] = method_summary
        rows.append(
            {
                "method": method,
                "seeds": "/".join(str(seed) for seed in seeds),
                "F_mean": fixed["F_score"]["mean"],
                "F_sd": fixed["F_score"]["sample_sd"],
                "AP_mean": swept["AP"]["mean"],
                "AP_sd": swept["AP"]["sample_sd"],
                "PRE_mean": fixed["PRE"]["mean"],
                "PRE_sd": fixed["PRE"]["sample_sd"],
                "REC_mean": fixed["REC"]["mean"],
                "REC_sd": fixed["REC"]["sample_sd"],
                "IoU_mean": fixed["IoU"]["mean"],
                "IoU_sd": fixed["IoU"]["sample_sd"],
                "parameters": method_summary["parameters"],
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
