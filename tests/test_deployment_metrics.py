import unittest

import torch

from litevilnet.metrics.deployment_metrics import BinarySegmentationMeter


def exact_reference(probabilities: torch.Tensor, target: torch.Tensor, thresholds: int):
    threshold_values = torch.linspace(0.0, 1.0, thresholds)
    best = None
    curve = []
    eps = 1e-7
    target = target.bool().reshape(-1)
    probabilities = probabilities.float().reshape(-1)
    for threshold in threshold_values:
        prediction = probabilities > threshold
        tp = int((prediction & target).sum())
        fp = int((prediction & ~target).sum())
        tn = int((~prediction & ~target).sum())
        fn = int((~prediction & target).sum())
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        fpr = fp / (fp + tn + eps)
        fnr = fn / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        curve.append((recall, precision))
        if best is None or f1 > best["MaxF"]:
            best = {
                "MaxF": f1,
                "BestThreshold": float(threshold),
                "PRE": precision,
                "REC": recall,
                "FPR": fpr,
                "FNR": fnr,
                "IoU": iou,
            }
    curve.sort()
    precisions = [point[1] for point in curve]
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    ap = 0.0
    previous_recall = 0.0
    for (recall, _), precision in zip(curve, precisions):
        ap += max(0.0, recall - previous_recall) * precision
        previous_recall = recall
    best["AP"] = ap
    return best


class BinarySegmentationMeterTest(unittest.TestCase):
    def test_histogram_implementation_matches_full_mask_reference(self):
        # Include threshold-equality and boundary cases, then random scores.
        generator = torch.Generator().manual_seed(20260723)
        probabilities = torch.cat(
            [
                torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0]),
                torch.rand(4091, generator=generator),
            ]
        ).reshape(1, 64, 64)
        target = torch.randint(0, 2, probabilities.shape, generator=generator).bool()
        meter = BinarySegmentationMeter(thresholds=11, max_pixels=10_000)
        meter.update(probabilities, target, input_type="prob")
        actual = meter.compute()
        expected = exact_reference(probabilities, target, thresholds=11)
        for key in ("MaxF", "AP", "PRE", "REC", "FPR", "FNR", "IoU", "BestThreshold"):
            self.assertAlmostEqual(actual[key], expected[key], places=7, msg=key)


if __name__ == "__main__":
    unittest.main()
