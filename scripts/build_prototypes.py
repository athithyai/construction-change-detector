"""Build the global K-means prototype corpus from all 2022 construction tiles.

Run this once after downloading tiles, before training.

Usage::

    python scripts/build_prototypes.py \\
        --tiles data/tiles/ \\
        --k 20 \\
        --out checkpoints/prototypes.pt

The script:
  1. Loads every (tile_2022, mask_2022) pair from the tile directory.
  2. Passes each tile through the frozen SAM2 image encoder.
  3. Extracts foreground feature vectors (where mask > 0).
  4. Runs KMeans(k) across all sites → k cluster centroids.
  5. Saves as a dict: {"prototypes": Tensor[k, 256], "k": int, "inertia": float}.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf

from data.dataset import ConstructionChangeDataset
from models.corpus_prototype import build_corpus_prototypes
from models.detector import ConstructionChangeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SAM2 prototype corpus from 2022 tiles.")
    p.add_argument("--tiles",   required=True,  help="Root tile directory.")
    p.add_argument("--k",       type=int, default=20, help="Number of prototype clusters.")
    p.add_argument("--out",     required=True,  help="Output path for prototypes.pt.")
    p.add_argument("--config",  default="configs/base.yaml")
    p.add_argument("--max-fg",  type=int, default=2000, help="Max foreground features sampled per site.")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load dataset (no augmentation, train split = all)
    dataset = ConstructionChangeDataset(
        tile_dir=args.tiles,
        image_size=cfg.data.image_size,
        augment=False,
        min_fg_pixels=cfg.data.min_fg_pixels,
    )
    logger.info("Dataset: %d samples", len(dataset))

    # Build model and load SAM2 encoder
    model = ConstructionChangeDetector(k_prototypes=args.k)
    model.load_sam2(cfg.model.sam2_hf_id)
    model.to(device)

    # Build prototypes
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    prototypes = build_corpus_prototypes(
        image_encoder=model.image_encoder,
        dataset=dataset,
        k=args.k,
        max_fg_per_site=args.max_fg,
        device=device,
        output_path=args.out,
    )
    logger.info("Prototypes shape: %s", prototypes.shape)
    logger.info("Saved to %s", args.out)


if __name__ == "__main__":
    main()
