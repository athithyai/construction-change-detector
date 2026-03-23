"""Score a geographic area and produce a probability GeoTIFF for interpreters.

The output is a 3-band float32 GeoTIFF in EPSG:28992 (RD New):
  Band 1 — combined construction probability   (load as heatmap in QGIS)
  Band 2 — preexisting sites probability       (was construction in 2022)
  Band 3 — new sites probability               (appeared after 2022)

Usage::

    python scripts/score_area.py \\
        --bbox "175000,445000,185000,455000" \\
        --prototypes checkpoints/prototypes.pt \\
        --checkpoint checkpoints/best.pt \\
        --mask2022 data/raw/construction_2022.gpkg \\
        --out outputs/construction_prob_2024.tif

The script:
  1. Downloads 2022 and 2024 PDOK tiles for the bounding box.
  2. Rasterises the 2022 construction polygons as a binary mask.
  3. Tiles the images into 1024×1024 patches (with overlap).
  4. Runs ConstructionChangeDetector on each tile pair.
  5. Stitches probability maps with Gaussian blending.
  6. Writes georeferenced multi-band GeoTIFF.

QGIS loading tip:
  Layer → Add Raster Layer → select the .tif
  Properties → Symbology → Singleband pseudocolor
  Band 1, min=0, max=1, colour ramp: transparent(0) → yellow(0.4) → red(1.0)
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from data.pdok_downloader import PDOKDownloader
from data.transforms import get_inference_transforms
from models.detector import ConstructionChangeDetector
from models.feature_utils import (
    load_geotiff_rgb,
    save_probability_geotiff,
    stitch_tiles,
    tile_image_for_inference,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score an area → probability GeoTIFF.")
    p.add_argument("--bbox",        required=True,
                   help="xmin,ymin,xmax,ymax in EPSG:28992 (RD New).")
    p.add_argument("--prototypes",  required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--mask2022",    required=True,
                   help="Path to 2022 construction polygons (.gpkg or .shp).")
    p.add_argument("--out",         required=True, help="Output GeoTIFF path.")
    p.add_argument("--config",      default="configs/inference.yaml")
    p.add_argument("--year-a",      type=int, default=2022)
    p.add_argument("--year-b",      type=int, default=2024)
    p.add_argument("--tile-size",   type=int, default=1024)
    p.add_argument("--overlap",     type=int, default=128)
    return p.parse_args()


def rasterise_polygons(
    gpkg_path: str,
    bbox: tuple,
    tile_px: int,
    crs: str = "EPSG:28992",
) -> np.ndarray:
    """Rasterise 2022 construction polygons within bbox → binary HxW float32."""
    import geopandas as gpd
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_bounds

    gdf = gpd.read_file(gpkg_path, bbox=bbox)
    if str(gdf.crs) != crs:
        gdf = gdf.to_crs(crs)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]

    transform = from_bounds(*bbox, width=tile_px, height=tile_px)
    if len(gdf) == 0:
        return np.zeros((tile_px, tile_px), dtype=np.float32)

    mask = rio_rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=(tile_px, tile_px),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask.astype(np.float32)


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    bbox = tuple(float(v) for v in args.bbox.split(","))
    assert len(bbox) == 4, "bbox must have 4 values: xmin,ymin,xmax,ymax"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | bbox: %s", device, bbox)

    # --- Load model ---
    model = ConstructionChangeDetector(k_prototypes=cfg.model.k_prototypes)
    model.load_sam2(cfg.model.sam2_hf_id)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_trainable_state_dict(state)
    model.to(device).eval()

    proto_ckpt = torch.load(args.prototypes, map_location=device)
    prototypes = proto_ckpt["prototypes"].to(device)

    transforms = get_inference_transforms(args.tile_size)

    # --- Download tiles ---
    with tempfile.TemporaryDirectory() as tmp:
        dl = PDOKDownloader(
            resolution=cfg.data.resolution,
            crs=cfg.data.crs,
        )
        path_a = str(Path(tmp) / f"tile_{args.year_a}.tif")
        path_b = str(Path(tmp) / f"tile_{args.year_b}.tif")

        logger.info("Downloading %d tile …", args.year_a)
        dl.download_tile(bbox, args.year_a, path_a)
        logger.info("Downloading %d tile …", args.year_b)
        dl.download_tile(bbox, args.year_b, path_b)

        img_a, src_a = load_geotiff_rgb(path_a)
        img_b, _     = load_geotiff_rgb(path_b)
        src_a.close()

        H, W = img_b.shape[:2]
        logger.info("Full image size: %dx%d", H, W)

        # --- Rasterise 2022 mask ---
        mask_2022_full = rasterise_polygons(args.mask2022, bbox, max(H, W), cfg.data.crs)
        mask_2022_full = mask_2022_full[:H, :W]

        # --- Tile ---
        tiles_b, coords = tile_image_for_inference(img_b, args.tile_size, args.overlap)
        tiles_a, _      = tile_image_for_inference(img_a, args.tile_size, args.overlap)
        masks_a, _      = tile_image_for_inference(
            (mask_2022_full * 255).astype(np.uint8)[:, :, None].repeat(3, 2),
            args.tile_size, args.overlap,
        )

        combined_tiles    = []
        preexisting_tiles = []
        new_tiles         = []

        for tile_b, tile_a, mask_tile in zip(tiles_b, tiles_a, masks_a):
            # Preprocess
            res_b = transforms(image=tile_b, image2=tile_a,
                                mask=np.zeros(tile_b.shape[:2], dtype=np.uint8),
                                mask2=np.zeros(tile_a.shape[:2], dtype=np.uint8))
            t_b = res_b["image"].unsqueeze(0).to(device)   # [1,3,H,W]
            t_a = res_b["image2"].unsqueeze(0).to(device)  # [1,3,H,W]

            # Binary mask at model resolution
            m_a = torch.from_numpy(
                (mask_tile[:, :, 0] > 127).astype(np.float32)
            ).unsqueeze(0).unsqueeze(0).to(device)          # [1,1,H,W]

            with torch.no_grad():
                out = model(t_a, t_b, m_a, prototypes)

            combined_tiles.append(
                out["prob_map"].squeeze().cpu().numpy()
            )
            preexisting_tiles.append(
                out["preexisting"].squeeze().cpu().numpy()
            )
            new_tiles.append(
                out["new_sites"].squeeze().cpu().numpy()
            )

        # --- Stitch ---
        shape = (H, W)
        prob_combined    = stitch_tiles(combined_tiles,    coords, shape)
        prob_preexisting = stitch_tiles(preexisting_tiles, coords, shape)
        prob_new         = stitch_tiles(new_tiles,         coords, shape)

        # --- Save ---
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save_probability_geotiff(
            prob_bands=[prob_combined, prob_preexisting, prob_new],
            band_names=[
                "combined_probability",
                "preexisting_construction",
                "new_construction",
            ],
            reference_path=path_b,
            output_path=args.out,
        )

    logger.info("Saved → %s", args.out)
    logger.info("Load in QGIS: Layer > Add Raster > select the .tif")
    logger.info("  Symbology: Singleband pseudocolor, Band 1, 0→transparent, 1→red")


if __name__ == "__main__":
    main()
