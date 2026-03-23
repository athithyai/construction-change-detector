"""Segmentation losses: BCE + Dice combined loss.

Dice loss is imbalance-resistant (construction sites occupy <10% of pixels),
so BCE alone would collapse to predicting all-background.  The combined loss
trains both heads jointly.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class BinaryDiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation (operates on logit inputs)."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits:  [B, 1, H, W] raw logits (no sigmoid applied).
            targets: [B, 1, H, W] binary float {0, 1}.
        """
        probs = torch.sigmoid(logits)
        B     = probs.shape[0]
        p_flat = probs.view(B, -1)
        t_flat = targets.view(B, -1)
        inter  = (p_flat * t_flat).sum(1)
        dice   = (2.0 * inter + self.smooth) / (
            p_flat.sum(1) + t_flat.sum(1) + self.smooth
        )
        return (1.0 - dice).mean()


class CombinedSegmentationLoss(nn.Module):
    """Weighted sum of BCE-with-logits and soft Dice loss.

    Args:
        bce_weight:  Weight for the BCE term.
        dice_weight: Weight for the Dice term.
        pos_weight:  Optional class-imbalance weight for BCE positive class.
                     Rule of thumb: background_pixels / foreground_pixels.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce  = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice = BinaryDiceLoss()

    def forward(self, logits: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            logits:  [B, 1, H, W] — raw model output (BEFORE sigmoid).
            targets: [B, 1, H, W] — binary ground-truth {0, 1}.

        Returns:
            Dict with keys: "loss", "bce_loss", "dice_loss".
        """
        bce_loss  = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total     = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return {
            "loss":      total,
            "bce_loss":  bce_loss.detach(),
            "dice_loss": dice_loss.detach(),
        }
