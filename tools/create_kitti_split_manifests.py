#!/usr/bin/env python3
"""Write reproducible historical or category-stratified KITTI manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from litevilnet.data.dataset import KITTIRoadDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default="configs/splits/kitti_road")
    parser.add_argument(
        "--strategy",
        choices=("historical-sorted", "stratified-random"),
        default="historical-sorted",
    )
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_stratified(samples: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["name"].split("_", 1)[0]].append(sample)

    target_train = int(len(samples) * 0.8)
    quotas = {category: int(len(group) * 0.8) for category, group in groups.items()}
    remaining = target_train - sum(quotas.values())
    fractional_order = sorted(
        groups,
        key=lambda category: (-(len(groups[category]) * 0.8 - quotas[category]), category),
    )
    for category in fractional_order[:remaining]:
        quotas[category] += 1

    rng = random.Random(seed)
    train, val = [], []
    for category in sorted(groups):
        group = sorted(groups[category], key=lambda sample: sample["name"])
        rng.shuffle(group)
        train.extend(group[: quotas[category]])
        val.extend(group[quotas[category] :])
    return sorted(train, key=lambda sample: sample["name"]), sorted(
        val, key=lambda sample: sample["name"]
    )


def main() -> None:
    args = parse_args()
    dataset = KITTIRoadDataset(
        args.data_root,
        split="train",
        use_augmentation=False,
    )
    # Recollect before the class-level split so the manifest generation itself
    # remains a transparent expression of the historical sorted 80/20 rule.
    all_samples = dataset._collect_samples()
    boundary = int(len(all_samples) * 0.8)
    if args.strategy == "historical-sorted":
        train_samples, val_samples = all_samples[:boundary], all_samples[boundary:]
    else:
        train_samples, val_samples = _split_stratified(all_samples, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": args.strategy,
        "seed": args.seed if args.strategy == "stratified-random" else None,
        "total": len(all_samples),
        "train": len(train_samples),
        "val": len(val_samples),
        "train_categories": dict(sorted(Counter(s["name"].split("_", 1)[0] for s in train_samples).items())),
        "val_categories": dict(sorted(Counter(s["name"].split("_", 1)[0] for s in val_samples).items())),
        "files": {},
    }
    for filename, samples in (("train.txt", train_samples), ("val.txt", val_samples)):
        text = "\n".join(sample["name"] for sample in samples) + "\n"
        (output_dir / filename).write_text(text, encoding="utf-8")
        payload["files"][filename] = {"sha256": _sha256_text(text), "entries": len(samples)}
    (output_dir / "manifest_metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
