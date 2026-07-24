#!/usr/bin/env python
"""Run one real ORFD optimization step and record CUDA memory feasibility."""

from __future__ import annotations

import argparse
import json

import torch
from torch.optim import AdamW

from litevilnet.data import get_orfd_dataloader, set_seed
from litevilnet.models.losses import VLLiNetLoss
from litevilnet.models.vllinet_ablation import get_ablation_model
from litevilnet.utils.common import system_metadata, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--config", default="full")
    parser.add_argument("--img_h", type=int, default=704)
    parser.add_argument("--img_w", type=int, default=1280)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--output", default="runs/revision_1/orfd_training_smoke.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    loader = get_orfd_dataloader(
        data_root=args.data_root,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_h=args.img_h,
        img_w=args.img_w,
        use_augmentation=True,
        seed=args.seed,
        drop_last=True,
    )
    batch = next(iter(loader))
    rgb = batch["rgb"].to(device, non_blocking=True)
    depth3 = batch["adi"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)
    model = get_ablation_model(args.config, pretrained=False).to(device).train()
    criterion = VLLiNetLoss(use_deep_supervision=model.use_deep_supervision).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda") if args.precision == "fp16" else None
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
        with torch.amp.autocast(device_type="cuda"):
            output = model(rgb, depth3, return_aux=True)
            logits, auxiliary = output if isinstance(output, tuple) else (output, None)
            loss = criterion(logits, auxiliary, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        output = model(rgb, depth3, return_aux=True)
        logits, auxiliary = output if isinstance(output, tuple) else (output, None)
        loss = criterion(logits, auxiliary, label)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    payload = {
        "status": "ok",
        "config": args.config,
        "dataset_samples": len(loader.dataset),
        "batch_size": args.batch_size,
        "input_shape_per_modality": list(rgb.shape),
        "label_shape": list(label.shape),
        "logit_shape": list(logits.shape),
        "auxiliary_outputs": len(auxiliary) if auxiliary is not None else 0,
        "precision": args.precision,
        "loss": float(loss.detach()),
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "cuda_peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
        "system": system_metadata(),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
