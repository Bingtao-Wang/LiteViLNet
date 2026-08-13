#!/usr/bin/env python3
"""Apply the completed SNE-RoadSeg ORFD summary to the two anonymous TeX files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def fmt(summary: dict, section: str, key: str) -> str:
    value = summary["methods"]["sne_roadseg"][section][key]
    return f"{100 * value['mean']:.2f}{{\\pm}}{100 * value['sample_sd']:.2f}"


def update_table(text: str, label: str, marker: str) -> str:
    values = [
        fmt(SUMMARY, "testing_fixed_argmax", "F_score"),
        fmt(SUMMARY, "testing_threshold_swept", "AP"),
        fmt(SUMMARY, "testing_fixed_argmax", "PRE"),
        fmt(SUMMARY, "testing_fixed_argmax", "REC"),
        fmt(SUMMARY, "testing_fixed_argmax", "IoU"),
    ]
    value_iter = iter(values)
    number_pattern = re.compile(r"\$[0-9.]+\{\\pm\}[0-9.]+\$")
    lines = text.splitlines(keepends=True)
    matches = 0
    for index, line in enumerate(lines):
        if marker not in line or "201.32" not in line:
            continue
        def replacement(_match: re.Match[str]) -> str:
            return "$" + next(value_iter) + "$"
        lines[index] = number_pattern.sub(replacement, line, count=5)
        matches += 1
        break
    if matches != 1:
        raise RuntimeError(f"Could not update {label}; expected one SNE ORFD row")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--root-tex", type=Path, required=True)
    parser.add_argument("--response-tex", type=Path, required=True)
    args = parser.parse_args()
    global SUMMARY
    SUMMARY = json.loads(args.summary.read_text(encoding="utf-8"))
    if SUMMARY.get("methods", {}).get("sne_roadseg", {}).get("seeds") != [40, 41, 42]:
        raise RuntimeError("Summary does not contain the formal SNE seeds 40, 41, 42")

    root = args.root_tex.read_text(encoding="utf-8")
    response = args.response_tex.read_text(encoding="utf-8")
    root = update_table(root, "root_1.tex Table II", r"\textcolor{cyan}{SNE-RoadSeg~\cite{sneroad}}")
    response = update_table(response, "ral_response_1.tex Table II", "SNE-RoadSeg (RGB+normal)")
    args.root_tex.write_text(root, encoding="utf-8")
    args.response_tex.write_text(response, encoding="utf-8")


if __name__ == "__main__":
    main()
