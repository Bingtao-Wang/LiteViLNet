#!/usr/bin/env python3
"""Validate and summarize the five matched RTX 4090 D FPS measurements."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sanitize_table1_supplement import sanitize


METHODS = ("litevilnet", "usnet", "sne_roadseg", "plard", "roadformer")
EXPECTED_PARAMETERS = {
    "litevilnet": 14_035_153,
    "usnet": 30_738_444,
    "sne_roadseg": 201_324_806,
    "plard": 76_929_142,
    "roadformer": 206_860_175,
}
EXPECTED_COMMITS = {
    "usnet": "d761158ad42df7dcb62fa257dd02ce11c85f94a5",
    "sne_roadseg": "5e7900bfd59887634ced687ffe85a73018a38659",
    "plard": "44485803092e729661c696ab6c03f6f2fabc8701",
    "roadformer": "f675a3467cb168ebc727648390c304279bbcb079",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--result-output-dir", type=Path)
    parser.add_argument("--anonymous-result-copies", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        path = args.input_root / f"{method}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("method") != method:
            raise ValueError(f"Method mismatch in {path}")
        records[method] = payload

    reference = records["litevilnet"]
    common_fields = (
        "protocol",
        "input_shape_rgb",
        "precision",
        "backend",
        "preprocessing_included",
        "host_to_device_transfer_included",
        "postprocessing_included",
        "warmup",
        "iterations_per_repeat",
        "repeats",
    )
    for method, payload in records.items():
        for field in common_fields:
            if payload[field] != reference[field]:
                raise ValueError(f"{method}: unmatched {field}: {payload[field]!r}")
        if payload["input_shape_rgb"] != [1, 3, 384, 1248]:
            raise ValueError(f"{method}: unexpected input shape")
        if payload["precision"] != "fp32" or payload["backend"] != "PyTorch":
            raise ValueError(f"{method}: expected PyTorch FP32")
        if payload["runtime"]["device"] != "NVIDIA GeForce RTX 4090 D":
            raise ValueError(f"{method}: unexpected GPU {payload['runtime']['device']}")
        if int(payload["parameters"]) != EXPECTED_PARAMETERS[method]:
            raise ValueError(f"{method}: unexpected parameter count {payload['parameters']}")
        if not re.fullmatch(r"[0-9a-f]{64}", payload["checkpoint_sha256"]):
            raise ValueError(f"{method}: invalid checkpoint SHA-256")
        if method != "litevilnet":
            source = payload.get("official_source")
            if not isinstance(source, dict) or source.get("commit") != EXPECTED_COMMITS[method]:
                raise ValueError(f"{method}: unexpected official source metadata")

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        payload = records[method]
        rows.append(
            {
                "method": method,
                "parameters_M": payload["parameters"] / 1e6,
                "mean_ms": payload["aggregate"]["mean_ms"],
                "sample_std_ms": payload["aggregate"]["sample_std_ms_across_repeat_means"],
                "fps": payload["aggregate"]["fps_from_mean"],
                "peak_allocated_cuda_MiB": payload["peak_allocated_cuda_mib"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
            }
        )
    summary = {
        "protocol": {
            field: reference[field] for field in common_fields
        },
        "device": reference["runtime"]["device"],
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.result_output_dir is not None:
        args.result_output_dir.mkdir(parents=True, exist_ok=True)
        for method, payload in records.items():
            output_payload = sanitize(payload) if args.anonymous_result_copies else payload
            if args.anonymous_result_copies:
                output_payload["supplement_anonymization"] = {
                    "local_absolute_paths": "replaced by portable placeholders or repository-relative paths",
                    "measurements_and_hashes_changed": False,
                }
            (args.result_output_dir / f"{method}.json").write_text(
                json.dumps(output_payload, indent=2) + "\n", encoding="utf-8"
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
