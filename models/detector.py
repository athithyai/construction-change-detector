"""ConstructionChangeDetector — top-level model.

Architecture:
  1. Shared frozen SAM2 Hiera+FPN image encoder encodes both the 2022 and 2024 tiles.
  2. AppearanceScorer:  cosine-similarity of 2024 features vs K corpus prototypes
                        → appear_map  [B, 1, H_f, W_f]
  3. ChangeScorer:      feature delta between 2022 and 2024 features
                        → change_map  [B, 1, H_f, W_f]
  4. Spatial routing via the rasterised 2022 construction mask:

       preexisting = appear_map  × mask_2022
       new_sites   = appear_map  × change_map × (1 − mask_2022)
       combined    = preexisting + new_sites          ← no pixel counted twice

  This ensures the change signal is NOT used for preexisting sites (where it
  would be low and would suppress valid detections) and IS required for new
  sites (to avoid flagging stable non-construction areas).

Trainable:   AppearanceScorer, ChangeScorer
Frozen:      SAM2 image encoder (Hiera backbone + FPN neck)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.appearance_scorer import AppearanceScorer
from models.change_scorer import ChangeScorer

logger = logging.getLogger(__name__)


class ConstructionChangeDetector(nn.Module):

    def __init__(self, k_prototypes: int = 20) -> None:
        super().__init__()

        # --- SAM2 image encoder (loaded lazily to avoid import side-effects) ---
        self.image_encoder: Optional[nn.Module] = None
        self._k = k_prototypes

        # --- Trainable heads ---
        self.appearance_scorer = AppearanceScorer(
            feature_dim=256, num_prototypes=k_prototypes
        )
        self.change_scorer = ChangeScorer(feature_dim=256)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def load_sam2(self, hf_id: str = "facebook/sam2.1-hiera-large") -> None:
        """Load the SAM2 image encoder from HuggingFace and freeze it.

        Call this once before training / inference.

        Args:
            hf_id: HuggingFace model ID for SAM2 (avoids local Hydra config).
        """
        from sam2.build_sam import build_sam2_hf  # type: ignore

        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device("cpu")
        logger.info("Loading SAM2 encoder from HuggingFace: %s (device=%s) …", hf_id, device)
        sam2 = build_sam2_hf(hf_id, device=str(device))
        self.sam2_model    = sam2          # full model — needed for SAM2ImagePredictor
        self.image_encoder = sam2.image_encoder

        # Freeze completely — only AppearanceScorer and ChangeScorer are trained
        for p in self.image_encoder.parameters():
            p.requires_grad_(False)
        self.image_encoder.eval()
        logger.info("SAM2 encoder frozen (%d parameters).",
                    sum(p.numel() for p in self.image_encoder.parameters()))

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image(self, image: Tensor) -> Tensor:
        """Encode a batch of images through the frozen SAM2 encoder.

        Args:
            image: [B, 3, H, W] float tensor, ImageNet-normalised.

        Returns:
            vision_features: [B, 256, H/16, W/16]
        """
        assert self.image_encoder is not None, (
            "Call load_sam2() before encode_image()."
        )
        out = self.image_encoder(image)
        return out["vision_features"]

    def forward(
        self,
        tile_2022: Tensor,      # [B, 3, H, W]
        tile_2024: Tensor,      # [B, 3, H, W]
        mask_2022: Tensor,      # [B, 1, H, W]  rasterised 2022 construction polygons
        prototypes: Tensor,     # [K, 256]       corpus prototypes (loaded from .pt)
    ) -> Dict[str, Tensor]:
        """Full forward pass.

        Returns a dict with three probability maps — all at the same spatial
        resolution as the input tiles [B, 1, H, W]:

        ``prob_map``    — combined detection (primary output for GeoTIFF export)
        ``preexisting`` — Path A: ongoing construction from 2022
        ``new_sites``   — Path B: construction that started after 2022
        """
        B, _, H, W = tile_2024.shape

        # Move prototypes to the same device
        prototypes = prototypes.to(tile_2024.device)

        # --- Encode (frozen, no gradients needed) ---
        feats_2022 = self.encode_image(tile_2022)   # [B, 256, H_f, W_f]
        feats_2024 = self.encode_image(tile_2024)   # [B, 256, H_f, W_f]
        _, _, Hf, Wf = feats_2022.shape

        # --- Downsample routing mask to feature resolution ---
        mask_feat = F.interpolate(
            mask_2022.float(), size=(Hf, Wf), mode="nearest"
        )  # [B, 1, H_f, W_f]  binary

        # --- Score heads ---
        appear = self.appearance_scorer(feats_2024, prototypes)   # [B,1,H_f,W_f]
        change = self.change_scorer(feats_2022, feats_2024)        # [B,1,H_f,W_f]

        # --- Spatial routing (disjoint paths, no pixel counted twice) ---
        preexisting = appear * mask_feat                           # PATH A
        new_sites   = appear * change * (1.0 - mask_feat)         # PATH B
        combined    = preexisting + new_sites                      # [B,1,H_f,W_f]

        # --- Upsample to input resolution ---
        def _up(x: Tensor) -> Tensor:
            return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        return {
            "prob_map":    _up(combined),     # primary
            "preexisting": _up(preexisting),  # band 2
            "new_sites":   _up(new_sites),    # band 3
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers (save / load only trainable weights)
    # ------------------------------------------------------------------

    def state_dict_trainable(self) -> dict:
        """State dict containing only trainable parameters."""
        return {
            "appearance_scorer": self.appearance_scorer.state_dict(),
            "change_scorer":     self.change_scorer.state_dict(),
        }

    def load_trainable_state_dict(self, state: dict) -> None:
        self.appearance_scorer.load_state_dict(state["appearance_scorer"])
        self.change_scorer.load_state_dict(state["change_scorer"])
