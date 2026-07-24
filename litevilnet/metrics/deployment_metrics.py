"""KITTI-style binary segmentation metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class BinarySegmentationMeter:
    """Collect binary segmentation metrics and threshold-swept MaxF."""

    def __init__(self, thresholds: int = 101, max_pixels: int = 8_000_000):
        self.threshold_values = torch.linspace(0.0, 1.0, thresholds)
        self.max_pixels = max_pixels
        # Bin predictions once during update instead of materializing one
        # full-resolution Boolean mask per threshold during compute().  Bin k
        # contains scores for which exactly k thresholds are strictly lower;
        # therefore reverse cumulative sums reproduce `prediction > threshold`
        # exactly, including scores equal to a threshold.
        self._positive_hist = torch.zeros(thresholds + 1, dtype=torch.int64)
        self._negative_hist = torch.zeros(thresholds + 1, dtype=torch.int64)
        self._count = 0

    def update(self, logits_or_probs: torch.Tensor, target: torch.Tensor, input_type: str = "logits") -> None:
        if logits_or_probs.dim() == 4:
            logits_or_probs = logits_or_probs.squeeze(1)
        if logits_or_probs.shape[-2:] != target.shape[-2:]:
            logits_or_probs = F.interpolate(
                logits_or_probs.unsqueeze(1),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        probs = torch.sigmoid(logits_or_probs) if input_type == "logits" else logits_or_probs
        probs = probs.detach().float().cpu().reshape(-1)
        target = target.detach().bool().cpu().reshape(-1)

        if probs.numel() > self.max_pixels:
            stride = max(1, probs.numel() // self.max_pixels)
            probs = probs[::stride]
            target = target[::stride]

        bucket = torch.bucketize(probs, self.threshold_values, right=False)
        self._positive_hist += torch.bincount(
            bucket[target], minlength=len(self.threshold_values) + 1
        )
        self._negative_hist += torch.bincount(
            bucket[~target], minlength=len(self.threshold_values) + 1
        )
        self._count += int(target.numel())

    def compute(self) -> dict[str, float]:
        if self._count == 0:
            return {
                "MaxF": 0.0,
                "AP": 0.0,
                "PRE": 0.0,
                "REC": 0.0,
                "FPR": 0.0,
                "FNR": 0.0,
                "BestThreshold": 0.5,
                "Precision": 0.0,
                "Recall": 0.0,
                "IoU": 0.0,
            }

        thresholds = self.threshold_values
        tp_by_threshold = torch.flip(
            torch.cumsum(torch.flip(self._positive_hist[1:], dims=[0]), dim=0),
            dims=[0],
        )
        fp_by_threshold = torch.flip(
            torch.cumsum(torch.flip(self._negative_hist[1:], dims=[0]), dim=0),
            dims=[0],
        )
        total_positive = int(self._positive_hist.sum().item())
        total_negative = int(self._negative_hist.sum().item())

        best = {
            "MaxF": -1.0,
            "BestThreshold": 0.5,
            "PRE": 0.0,
            "REC": 0.0,
            "FPR": 0.0,
            "FNR": 0.0,
            "IoU": 0.0,
        }
        curve: list[tuple[float, float]] = []
        eps = 1e-7
        for index, threshold in enumerate(thresholds):
            tp = int(tp_by_threshold[index].item())
            fp = int(fp_by_threshold[index].item())
            tn = total_negative - fp
            fn = total_positive - tp
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            fpr = fp / (fp + tn + eps)
            fnr = fn / (tp + fn + eps)
            f1 = 2.0 * precision * recall / (precision + recall + eps)
            iou = tp / (tp + fp + fn + eps)
            curve.append((recall, precision))
            if f1 > best["MaxF"]:
                best = {
                    "MaxF": f1,
                    "BestThreshold": float(threshold.item()),
                    "PRE": precision,
                    "REC": recall,
                    "FPR": fpr,
                    "FNR": fnr,
                    "IoU": iou,
                }
        curve.sort()
        ap = 0.0
        if curve:
            recalls = [point[0] for point in curve]
            precisions = [point[1] for point in curve]
            for idx in range(len(precisions) - 2, -1, -1):
                precisions[idx] = max(precisions[idx], precisions[idx + 1])
            prev_recall = 0.0
            for recall, precision in zip(recalls, precisions):
                ap += max(0.0, recall - prev_recall) * precision
                prev_recall = recall

        best["AP"] = ap
        best["Precision"] = best["PRE"]
        best["Recall"] = best["REC"]
        return best
