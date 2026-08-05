#!/usr/bin/env python3
"""Aggregate matched-protocol baseline seeds into auditable JSON and CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sanitize_table1_supplement import sanitize


METRICS = ("MaxF", "AP", "PRE", "REC", "FPR", "FNR", "IoU", "BestThreshold")
PROTOCOL_FIELDS = ("input_size", "train_count", "val_count", "epochs_requested", "batch_size", "amp")
EXPECTED_COMMON_PROTOCOL = {
    "input_size": [384, 1248],
    "train_count": 231,
    "val_count": 58,
    "epochs_requested": 150,
}
EXPECTED_METHOD_PROTOCOL = {
    "usnet": {"batch_size": 2, "amp": True},
    "sne_roadseg": {"batch_size": 2, "amp": True},
    "plard": {"batch_size": 4, "amp": False},
    "roadformer": {"batch_size": 4, "amp": False},
    "offnet": {"batch_size": 2, "amp": False},
}
EXPECTED_SOURCES = {
    "usnet": (
        "https://github.com/morancyc/USNet.git",
        "d761158ad42df7dcb62fa257dd02ce11c85f94a5",
    ),
    "sne_roadseg": (
        "https://github.com/hlwang1124/SNE-RoadSeg.git",
        "5e7900bfd59887634ced687ffe85a73018a38659",
    ),
    "plard": (
        "https://github.com/zhechen/PLARD.git",
        "44485803092e729661c696ab6c03f6f2fabc8701",
    ),
    "roadformer": (
        "https://github.com/LiJiahang617/Road-Former.git",
        "f675a3467cb168ebc727648390c304279bbcb079",
    ),
    "offnet": (
        "https://github.com/chaytonmin/Off-Road-Freespace-Detection",
        "50e63d24836198e8fb5af707e521f414104b4876",
    ),
}
EXPECTED_SPLIT_HASHES = (
    "93a8b849a531e9bd938c65120816f5ad4bd62f563e7f0d68ac6c0e6046425867",
    "69b10e5ff641d5cea81d2f0832ada2c31ee5f3b3f8ced9e4e962f889a722976f",
)
SPLIT_PATHS = (
    REPO_ROOT / "configs" / "splits" / "kitti_road" / "stratified_seed20260723" / "train.txt",
    REPO_ROOT / "configs" / "splits" / "kitti_road" / "stratified_seed20260723" / "val.txt",
)
PROVENANCE_PATH = (
    REPO_ROOT / "docs" / "ral" / "table1_matched_baselines" / "source_provenance.json"
)
PROVENANCE_NAMES = {
    "usnet": "USNet",
    "sne_roadseg": "SNE-RoadSeg",
    "plard": "PLARD",
    "roadformer": "RoadFormer",
    "offnet": "OFF-Net",
}
EXPECTED_RECORDED_FILES = {
    "usnet": {"model/usnet.py", "loss.py"},
    "sne_roadseg": {
        "models/networks.py",
        "models/roadseg_model.py",
    },
    "plard": {
        "ptsemseg/models/plard.py",
        "ptsemseg/loss.py",
    },
    "roadformer": {
        "configs/roadformer_kitti/roadformer_convnext-b_kitti-384x1280.py",
        "mmpretrain_custom/models/backbones/twin_convnext.py",
        "mmdet_custom/models/layers/roadformer_pixel_decoder.py",
    },
    "offnet": {
        "models/transformer_models/backbones/transformer.py",
        "models/transformer_models/decode_heads/head.py",
        "models/loss.py",
    },
}
EXPECTED_PARAMETERS = {
    "usnet": 30_738_444,
    "sne_roadseg": 201_324_806,
    "plard": 76_929_142,
    "roadformer": 206_860_175,
    "offnet": 25_209_608,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed-output-dir", type=Path)
    parser.add_argument("--expected-seeds", default="40,41,42")
    parser.add_argument(
        "--anonymous-seed-copies",
        action="store_true",
        help="Replace local paths in copied seed JSON; raw run JSON remains untouched",
    )
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accepted_source_hashes(
    provenance: dict[str, Any], baseline: str
) -> dict[str, set[str]]:
    """Return accepted fresh-clone and formal-run hashes for official files."""

    method = provenance[PROVENANCE_NAMES[baseline]]
    return {
        relative_path: {
            entry["canonical_lf_sha256"],
            entry["working_file_sha256"],
        }
        for relative_path, entry in method["files"].items()
    }


def main() -> None:
    args = parse_args()
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    provenance_split_hashes = (
        provenance["inputs"]["train_manifest_sha256"],
        provenance["inputs"]["val_manifest_sha256"],
    )
    if provenance_split_hashes != EXPECTED_SPLIT_HASHES:
        raise ValueError("source_provenance.json records unexpected split hashes")
    local_split_hashes = tuple(file_sha256(path) for path in SPLIT_PATHS)
    if local_split_hashes != EXPECTED_SPLIT_HASHES:
        raise ValueError(f"versioned split manifest hash mismatch: {local_split_hashes}")
    expected_seeds = sorted(int(value) for value in args.expected_seeds.split(",") if value)
    grouped: dict[str, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for path in sorted(args.input_root.glob("*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != "matched local KITTI perspective-view retraining":
            continue
        grouped[payload["baseline"]].append((payload, path))
    if not grouped:
        raise FileNotFoundError(f"No matched result.json files under {args.input_root}")
    if set(grouped) != set(EXPECTED_SOURCES):
        raise ValueError(
            "Matched summary requires exactly "
            f"{sorted(EXPECTED_SOURCES)}, found {sorted(grouped)}"
        )

    groups: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for baseline, records in sorted(grouped.items()):
        if baseline not in EXPECTED_SOURCES:
            raise ValueError(f"Unexpected baseline in matched summary: {baseline}")
        records.sort(key=lambda item: int(item[0]["seed"]))
        seeds = [int(payload["seed"]) for payload, _ in records]
        if seeds != expected_seeds:
            raise ValueError(f"{baseline}: expected seeds {expected_seeds}, found {seeds}")
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{baseline}: duplicate seeds {seeds}")
        for field in PROTOCOL_FIELDS:
            values = {json.dumps(payload[field], sort_keys=True) for payload, _ in records}
            if len(values) != 1:
                raise ValueError(f"{baseline}: protocol mismatch in {field}: {values}")
            expected_value = (
                EXPECTED_COMMON_PROTOCOL[field]
                if field in EXPECTED_COMMON_PROTOCOL
                else EXPECTED_METHOD_PROTOCOL[baseline][field]
            )
            if records[0][0][field] != expected_value:
                raise ValueError(
                    f"{baseline}: expected {field}={expected_value!r}, "
                    f"found {records[0][0][field]!r}"
                )
        split_hashes = {
            (
                payload["split_metadata"]["train_manifest_sha256"],
                payload["split_metadata"]["val_manifest_sha256"],
            )
            for payload, _ in records
        }
        if len(split_hashes) != 1:
            raise ValueError(f"{baseline}: split hash mismatch")
        if split_hashes != {EXPECTED_SPLIT_HASHES}:
            raise ValueError(
                f"{baseline}: expected split hashes {EXPECTED_SPLIT_HASHES}, "
                f"found {split_hashes}"
            )
        commits = {payload["official_commit"] for payload, _ in records}
        remotes = {payload["official_remote"] for payload, _ in records}
        expected_remote, expected_commit = EXPECTED_SOURCES[baseline]
        if commits != {expected_commit}:
            raise ValueError(f"{baseline}: expected official commit {expected_commit}, found {commits}")
        if remotes != {expected_remote}:
            raise ValueError(f"{baseline}: expected official remote {expected_remote}, found {remotes}")
        if any(payload.get("official_semantic_diff_clean") is False for payload, _ in records):
            raise ValueError(f"{baseline}: a run records a semantically modified official source")
        source_hash_sets = {
            json.dumps(payload["official_file_sha256"], sort_keys=True) for payload, _ in records
        }
        if len(source_hash_sets) != 1:
            raise ValueError(f"{baseline}: official source-file hash mismatch")
        expected_hashes = accepted_source_hashes(provenance, baseline)
        recorded_hashes = records[0][0]["official_file_sha256"]
        recorded_file_set = set(recorded_hashes)
        if not EXPECTED_RECORDED_FILES[baseline].issubset(recorded_file_set):
            raise ValueError(
                f"{baseline}: imported official source-file set omits a required core file"
            )
        if not recorded_file_set.issubset(expected_hashes):
            raise ValueError(f"{baseline}: unrecognized official source file in result metadata")
        for relative_path, recorded_hash in recorded_hashes.items():
            if recorded_hash not in expected_hashes[relative_path]:
                raise ValueError(
                    f"{baseline}: unrecognized official source hash for {relative_path}: "
                    f"{recorded_hash}"
                )
        for payload, _ in records:
            if payload["split_metadata"].get("train_val_overlap") != 0:
                raise ValueError(f"{baseline}: nonzero train/validation overlap")
            normal_metadata = payload.get("normal_cache_metadata")
            if baseline in {"usnet", "sne_roadseg"}:
                if not isinstance(normal_metadata, dict):
                    raise ValueError(f"{baseline}: missing official SNE cache metadata")
                if normal_metadata.get("official_commit") != EXPECTED_SOURCES["sne_roadseg"][1]:
                    raise ValueError(f"{baseline}: unexpected official SNE cache commit")
                sne_hashes = accepted_source_hashes(provenance, "sne_roadseg")["models/sne_model.py"]
                if normal_metadata.get("official_sne_sha256") not in sne_hashes:
                    raise ValueError(f"{baseline}: unexpected official SNE cache source hash")
            elif baseline == "offnet":
                if not isinstance(normal_metadata, dict):
                    raise ValueError("offnet: missing OFF-Net SNE cache metadata")
                if normal_metadata.get("profile") != "offnet":
                    raise ValueError("offnet: normal cache used the wrong SNE profile")
                if normal_metadata.get("official_commit") != EXPECTED_SOURCES["offnet"][1]:
                    raise ValueError("offnet: unexpected SNE cache commit")
                offnet_sne_hashes = accepted_source_hashes(provenance, "offnet")[
                    "models/sne_model.py"
                ]
                if normal_metadata.get("official_sne_sha256") not in offnet_sne_hashes:
                    raise ValueError("offnet: unexpected SNE cache source hash")
            checkpoint_hash = payload.get("best_checkpoint_sha256")
            if not isinstance(checkpoint_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_hash):
                raise ValueError(f"{baseline}: invalid checkpoint SHA-256 for seed {payload['seed']}")
            checkpoint_value = payload.get("best_checkpoint")
            if not isinstance(checkpoint_value, str) or not checkpoint_value:
                raise ValueError(f"{baseline}: missing checkpoint path for seed {payload['seed']}")
            checkpoint_path = Path(checkpoint_value)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"{baseline}: checkpoint missing for seed {payload['seed']}: {checkpoint_path}"
                )
            if file_sha256(checkpoint_path) != checkpoint_hash:
                raise ValueError(f"{baseline}: checkpoint SHA-256 mismatch for seed {payload['seed']}")
            best_epoch = int(payload["best_epoch"])
            if not 1 <= best_epoch <= int(payload["epochs_requested"]):
                raise ValueError(f"{baseline}: invalid best epoch {best_epoch} for seed {payload['seed']}")
        parameters = {int(payload["parameters"]) for payload, _ in records}
        if len(parameters) != 1:
            raise ValueError(f"{baseline}: parameter mismatch: {parameters}")
        if parameters != {EXPECTED_PARAMETERS[baseline]}:
            raise ValueError(
                f"{baseline}: expected {EXPECTED_PARAMETERS[baseline]} parameters, "
                f"found {parameters}"
            )

        metric_summary = {
            metric: stats([float(payload["metrics"][metric]) for payload, _ in records])
            for metric in METRICS
        }
        group = {
            "baseline": baseline,
            "seeds": seeds,
            "parameters": parameters.pop(),
            "official_remote": expected_remote,
            "official_commit": commits.pop(),
            "official_file_sha256": recorded_hashes,
            "official_sne_sha256": (
                records[0][0]["normal_cache_metadata"]["official_sne_sha256"]
                if baseline in {"usnet", "sne_roadseg", "offnet"}
                else None
            ),
            "protocol": {field: records[0][0][field] for field in PROTOCOL_FIELDS},
            "train_manifest_sha256": records[0][0]["split_metadata"]["train_manifest_sha256"],
            "val_manifest_sha256": records[0][0]["split_metadata"]["val_manifest_sha256"],
            "metrics": metric_summary,
            "checkpoints": {
                str(payload["seed"]): payload["best_checkpoint_sha256"] for payload, _ in records
            },
            "source_files": [str(path.relative_to(args.input_root)) for _, path in records],
        }
        groups.append(group)
        row: dict[str, Any] = {
            "baseline": baseline,
            "seeds": ",".join(str(seed) for seed in seeds),
            "parameters_M": group["parameters"] / 1e6,
            "official_commit": group["official_commit"],
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = metric_summary[metric]["mean"]
            row[f"{metric}_sample_std"] = metric_summary[metric]["sample_std"]
        csv_rows.append(row)

    output = {
        "protocol": "same 231/58 category-stratified KITTI split, 384x1248 input, 101-threshold pixel-accumulated perspective-view evaluation",
        "groups": groups,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(csv_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    if args.seed_output_dir is not None:
        args.seed_output_dir.mkdir(parents=True, exist_ok=True)
        for baseline, records in grouped.items():
            for payload, _ in records:
                output_path = args.seed_output_dir / f"{baseline}_seed{payload['seed']}.json"
                output_payload = sanitize(payload) if args.anonymous_seed_copies else payload
                if args.anonymous_seed_copies:
                    output_payload["supplement_anonymization"] = {
                        "local_absolute_paths": "replaced by portable placeholders or repository-relative paths",
                        "metrics_and_hashes_changed": False,
                    }
                output_path.write_text(
                    json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
