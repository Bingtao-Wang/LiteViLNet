#!/usr/bin/env python
"""Generate RA-L deployment figures from benchmark CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RA-L deployment results")
    parser.add_argument("--benchmark_csv", default="runs/benchmark/benchmark_summary.csv")
    parser.add_argument("--eval_csv", default="runs/eval/eval_summary.csv")
    parser.add_argument("--output_dir", default="docs/ral/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if Path(args.benchmark_csv).exists() and Path(args.eval_csv).exists():
        bench = pd.read_csv(args.benchmark_csv)
        metrics = pd.read_csv(args.eval_csv)
        merged = bench.merge(metrics[["preset", "MaxF"]], on="preset", how="left")
        merged = merged.dropna(subset=["fps", "MaxF"])
        if not merged.empty:
            plt.figure(figsize=(6.2, 4.0))
            for _, row in merged.iterrows():
                plt.scatter(row["mean_ms"], row["MaxF"] * 100.0, s=80)
                plt.text(row["mean_ms"], row["MaxF"] * 100.0, str(row["model"]), fontsize=8)
            plt.xlabel("Latency (ms, lower is better)")
            plt.ylabel("MaxF (%)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "pareto_maxf_latency.pdf")
            plt.savefig(output_dir / "pareto_maxf_latency.png", dpi=300)

    if Path(args.benchmark_csv).exists():
        bench = pd.read_csv(args.benchmark_csv)
        if {"model", "backend", "precision", "fps"}.issubset(bench.columns):
            labels = bench["model"].astype(str) + "\n" + bench["backend"].astype(str) + "-" + bench["precision"].astype(str)
            plt.figure(figsize=(7.0, 4.0))
            plt.bar(labels, bench["fps"])
            plt.ylabel("FPS")
            plt.xticks(rotation=30, ha="right")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "jetson_fps_bar.pdf")
            plt.savefig(output_dir / "jetson_fps_bar.png", dpi=300)

    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
