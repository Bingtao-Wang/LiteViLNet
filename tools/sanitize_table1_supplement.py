#!/usr/bin/env python3
"""Create anonymous result copies and reject identity/path leakage.

The formal run JSON remains untouched on the host.  Only the temporary copies
used by ``package_table1_matched_baselines.sh`` are rewritten.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


LEAK_PATTERNS = {
    "Linux home path": re.compile(r"/(?:home|Users|root)/[^\s\"']+"),
    "local data hierarchy": re.compile("/data/" + r"Database/[^\s\"']*"),
    "ephemeral local path": re.compile(r"/(?:dev/shm|tmp)/[^\s\"']+"),
    "Windows drive path": re.compile(r"\b[A-Z]:\\[^\s\"']+", re.IGNORECASE),
    "SSH Git remote": re.compile(r"(?<![\w.-])git" + r"@[^\s\"']+"),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path)
    parser.add_argument("--output-results", type=Path)
    parser.add_argument("--scan-root", type=Path)
    parser.add_argument(
        "--deny-token",
        action="append",
        default=[],
        help="Additional identity token to reject; may be supplied repeatedly",
    )
    return parser.parse_args()


def anonymize_string(value: str) -> str:
    if not value.startswith("/"):
        sanitized = value
    elif "SNE-RoadSeg/source" in value or value.endswith("/SNE-RoadSeg"):
        sanitized = "${SNE_SOURCE}"
    elif "USNet/source" in value or value.endswith("/USNet"):
        sanitized = "${USNET_SOURCE}"
    elif value.endswith("/PLARD"):
        sanitized = "${PLARD_SOURCE}"
    elif value.endswith("/Road-Former"):
        sanitized = "${ROADFORMER_SOURCE}"
    elif value.endswith("/Final_Dataset"):
        sanitized = "${ORFD_ROOT}"
    elif value.endswith("/velodyne_extracted"):
        sanitized = "${VELODYNE_ROOT}"
    elif value.endswith("/robot_road_raw"):
        sanitized = "${ROBOT_ROOT}"
    elif "/revision_1_runs/" in value:
        sanitized = "runs/revision_1/" + value.split("/revision_1_runs/", 1)[1]
    else:
        sanitized = ""
        for marker in ("/configs/", "/runs/", "/tools/", "/docs/", "/litevilnet/"):
            if marker in value:
                sanitized = marker[1:] + value.split(marker, 1)[1]
                break
        if not sanitized and value.endswith("/depth_u16"):
            sanitized = "${DEPTH_ROOT}/depth_u16"
        elif not sanitized and value.endswith("/kitti_road"):
            sanitized = "${KITTI_ROOT}"
        elif not sanitized:
            sanitized = "${ANONYMIZED_LOCAL_ROOT}/" + Path(value).name

    for label in (
        "Linux home path",
        "local data hierarchy",
        "ephemeral local path",
        "Windows drive path",
    ):
        sanitized = LEAK_PATTERNS[label].sub("${ANONYMIZED_LOCAL_PATH}", sanitized)
    sanitized = LEAK_PATTERNS["email address"].sub("anonymous-email", sanitized)
    sanitized = LEAK_PATTERNS["SSH Git remote"].sub("anonymous-git-remote", sanitized)
    return sanitized


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "anonymous-host"
            if key.lower() in {"hostname", "host_name", "machine_name", "username", "user"}
            and isinstance(item, str)
            else sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return anonymize_string(value)
    return value


def sanitize_results(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing non-empty staging target: {output}")
    output.mkdir(parents=True)
    for source_path in sorted(source.rglob("*")):
        if not source_path.is_file():
            continue
        relative = source_path.relative_to(source)
        output_path = output / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.suffix == ".json":
            payload = sanitize(json.loads(source_path.read_text(encoding="utf-8")))
            if isinstance(payload, dict):
                payload["supplement_anonymization"] = {
                    "local_absolute_paths": "replaced by portable placeholders or repository-relative paths",
                    "metrics_and_hashes_changed": False,
                }
            output_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        else:
            shutil.copyfile(source_path, output_path)


def scan_tree(root: Path, deny_tokens: tuple[str, ...] = ()) -> None:
    patterns = dict(LEAK_PATTERNS)
    for index, token in enumerate(deny_tokens):
        if token:
            patterns[f"supplied deny token {index + 1}"] = re.compile(re.escape(token), re.IGNORECASE)
    leaks: list[str] = []
    for path in sorted(root.rglob("*")):
        relative_path = str(path.relative_to(root))
        for label, pattern in patterns.items():
            match = pattern.search(relative_path)
            if match:
                leaks.append(f"{relative_path}: filename {label}: {match.group(0)}")
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in patterns.items():
            match = pattern.search(content)
            if match:
                leaks.append(f"{relative_path}: {label}: {match.group(0)}")
    if leaks:
        raise RuntimeError("Anonymous supplement scan failed:\n" + "\n".join(leaks))


def main() -> None:
    args = parse_args()
    if (args.source_results is None) != (args.output_results is None):
        raise ValueError("--source-results and --output-results must be supplied together")
    if args.source_results is not None:
        sanitize_results(args.source_results, args.output_results)
    if args.scan_root is not None:
        scan_tree(args.scan_root, tuple(args.deny_token))
    if args.source_results is None and args.scan_root is None:
        raise ValueError("No action requested")


if __name__ == "__main__":
    main()
