"""ConstructionChangeDataset — loads (tile_2022, tile_2024, mask_2022, mask_2024) pairs.

Directory layout expected (produced by PDOKDownloader)::

    tile_dir/
      <site_id>/
        tile_2022.tif
        tile_2024.tif
        mask_2022.png    ← binary mask from 2022 construction polygon
        mask_2024.png    ← optional; falls back to mask_2022 if absent
        meta.json        ← {"sample_type": "preexisting"|"new"|"negative"}

sample_type meanings:
  preexisting — construction existed in 2022, may still be active in 2024
  new         — no construction in 2022 tile, but construction appeared by 2024
  negative    — non-construction tile (farmland, water, road, etc.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from data.transforms import get_train_transforms, get_inference_transforms

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

logger = logging.getLogger(__name__)


class ConstructionChangeDataset(Dataset):
    """Paired (2022, 2024) aerial tile dataset for construction change detection."""

    def __init__(
        self,
        tile_dir: str,
        image_size: int = 1024,
        augment: bool = True,
        min_fg_pixels: int = 500,
        split: Optional[str] = None,
        val_split: float = 0.15,
        seed: int = 42,
    ) -> None:
        """
        Args:
            tile_dir:       Root directory of downloaded tiles.
            image_size:     Spatial size fed to the model.
            augment:        If True use training augmentations, else inference-only.
            min_fg_pixels:  Skip samples whose mask_2022 has fewer foreground px.
            split:          "train" | "val" | None (use all).
            val_split:      Fraction of sites to hold out for validation.
            seed:           Random seed for the train/val split.
        """
        self.tile_dir   = Path(tile_dir)
        self.image_size = image_size
        self.augment    = augment
        self.transforms = (
            get_train_transforms(image_size) if augment
            else get_inference_transforms(image_size)
        )

        all_samples = self._scan(min_fg_pixels)

        if split is not None:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(all_samples))
            n_val   = max(1, int(len(all_samples) * val_split))
            val_idx = set(indices[:n_val].tolist())
            if split == "val":
                all_samples = [all_samples[i] for i in val_idx]
            else:
                all_samples = [s for i, s in enumerate(all_samples) if i not in val_idx]

        self.samples: List[Dict] = all_samples
        logger.info("Dataset (%s split): %d samples", split or "all", len(self.samples))

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        tile_2024 = self._load_image(s["tile_b"])   # query year
        tile_2022 = self._load_image(s["tile_a"])   # reference year
        mask_2022 = self._load_mask(s["mask_a"])
        mask_2024 = self._load_mask(s["mask_b"])

        result = self.transforms(
            image=tile_2024,
            image2=tile_2022,
            mask=mask_2024,
            mask2=mask_2022,
        )

        return {
            "tile_2024":   result["image"],                                       # [3,H,W]
            "tile_2022":   result["image2"],                                      # [3,H,W]
            "mask_2024":   result["mask"].unsqueeze(0).float() / 255.0,           # [1,H,W] {0,1}
            "mask_2022":   result["mask2"].unsqueeze(0).float() / 255.0,          # [1,H,W] {0,1}
            "sample_type": s["sample_type"],
            "site_id":     s["site_id"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan(self, min_fg_pixels: int) -> List[Dict]:
        samples = []
        for site_dir in sorted(self.tile_dir.iterdir()):
            if not site_dir.is_dir():
                continue

            meta_path = site_dir / "meta.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)

            year_a = meta.get("year_a", 2022)
            year_b = meta.get("year_b", 2024)

            tile_a = site_dir / f"tile_{year_a}.tif"
            tile_b = site_dir / f"tile_{year_b}.tif"
            mask_a = site_dir / f"mask_{year_a}.png"
            # mask_2024 is optional — fall back to mask_2022 for preexisting sites
            mask_b = site_dir / f"mask_{year_b}.png"
            if not mask_b.exists():
                mask_b = mask_a

            if not (tile_a.exists() and tile_b.exists() and mask_a.exists()):
                continue

            # Skip tiny masks
            mask_arr = np.array(Image.open(mask_a))
            if mask_arr.sum() < min_fg_pixels * 255:
                logger.debug("Skipping %s — mask too small", site_dir.name)
                continue

            samples.append({
                "site_id":     site_dir.name,
                "tile_a":      str(tile_a),
                "tile_b":      str(tile_b),
                "mask_a":      str(mask_a),
                "mask_b":      str(mask_b),
                "sample_type": meta.get("sample_type", "preexisting"),
            })
        return samples

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        """Load image as HxWx3 uint8 numpy array.

        Handles both standard images (.jpg, .png) and GeoTIFFs (.tif).
        """
        if path.endswith(".tif") or path.endswith(".tiff"):
            if _HAS_RASTERIO:
                with rasterio.open(path) as src:
                    # Read first 3 bands (RGB)
                    n = min(src.count, 3)
                    bands = src.read(list(range(1, n + 1)))  # [C, H, W]
                    img = np.transpose(bands, (1, 2, 0))      # [H, W, C]
                    if img.dtype != np.uint8:
                        img = np.clip(img, 0, 255).astype(np.uint8)
                    return img
        return np.array(Image.open(path).convert("RGB"))

    @staticmethod
    def _load_mask(path: str) -> np.ndarray:
        """Load binary mask as HxW uint8 (values 0 or 255)."""
        mask = np.array(Image.open(path).convert("L"))
        return (mask > 127).astype(np.uint8) * 255
