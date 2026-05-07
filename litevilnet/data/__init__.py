"""Dataset and training helpers for LiteViLNet."""

from .dataset import KITTIRoadDataset, get_dataloader
from litevilnet.metrics.road_metrics import RoadMetrics

import torch


class AverageMeter:
    """Track a running average."""

    def __init__(self, name: str = "", fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def print_metrics(metrics_dict, prefix: str = "") -> None:
    print(f"{prefix}metrics:")
    print("-" * 50)
    for key, value in metrics_dict.items():
        print(f"  {key}: {value}")
    print("-" * 50)


def format_metrics(metrics_dict) -> str:
    items = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics_dict.items()]
    return " | ".join(items)


def save_checkpoint(model, optimizer, epoch, best_metric, path, args=None, ema_state_dict=None) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }
    if args is not None:
        checkpoint["args"] = vars(args) if hasattr(args, "__dict__") else args
    if ema_state_dict is not None:
        checkpoint["ema_state_dict"] = ema_state_dict
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, strict: bool = True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_metric", 0)


def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class EarlyStopping:
    """Simple early stopping helper."""

    def __init__(self, patience: int = 20, min_delta: float = 0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = score > self.best_score + self.min_delta if self.mode == "max" else score < self.best_score - self.min_delta
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


__all__ = [
    "KITTIRoadDataset",
    "get_dataloader",
    "RoadMetrics",
    "AverageMeter",
    "print_metrics",
    "format_metrics",
    "save_checkpoint",
    "load_checkpoint",
    "set_seed",
    "count_parameters",
    "get_lr",
    "EarlyStopping",
]
