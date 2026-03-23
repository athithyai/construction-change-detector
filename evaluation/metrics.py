"""Segmentation evaluation metrics: IoU, F1, precision, recall.

SegmentationEvaluator accumulates pixel-level TP/FP/FN counts across batches
and computes metrics at the end of an epoch.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

import torch
from torch import Tensor


class SegmentationEvaluator:
    """Online accumulator for binary segmentation metrics.

    Usage::

        ev = SegmentationEvaluator(threshold=0.5)
        for batch in loader:
            preds = model(...)["prob_map"]
            ev.update(preds, batch["mask_2024"])
        metrics = ev.compute()
        ev.reset()
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._tp = 0
        self._fp = 0
        self._fn = 0

    def update(self, prob_map: Tensor, gt_mask: Tensor) -> None:
        """Accumulate counts for one batch.

        Args:
            prob_map: [B, 1, H, W] probability values in [0, 1].
            gt_mask:  [B, 1, H, W] binary ground-truth {0, 1}.
        """
        pred = (prob_map >= self.threshold).float()
        gt   = (gt_mask  >= 0.5).float()

        self._tp += int((pred * gt).sum().item())
        self._fp += int((pred * (1 - gt)).sum().item())
        self._fn += int(((1 - pred) * gt).sum().item())

    def compute(self) -> Dict[str, float]:
        """Return all metrics as a plain dict."""
        tp, fp, fn = self._tp, self._fp, self._fn
        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        f1        = 2 * precision * recall / (precision + recall + 1e-6)
        iou       = tp / (tp + fp + fn + 1e-6)
        return {
            "iou":       round(iou, 4),
            "f1":        round(f1,  4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
        }
