"""Score a geographic area and produce probability outputs in EPSG:28992 (RD New).

Outputs
-------
GeoTIFF  (--out *.tif)
    3-band float32 raster:
      Band 1 — combined construction probability
      Band 2 — preexisting sites probability  (was construction in 2022)
      Band 3 — new sites probability          (appeared after 2022)

GeoPackage  (--out-gpkg *.gpkg)
    Vector polygons of detected construction sites with attributes:
      type             — "preexisting" | "new"
      prob_combined    — mean combined probability within polygon
      prob_preexisting — mean preexisting probability
      prob_new         — mean new-site probability
      area_m2          — polygon area in square metres

Usage::

    python scripts/score_area.py \\
        --bbox "175000,445000,185000,455000" \\
        --prototypes checkpoints/prototypes.pt \\
        --checkpoint checkpoints/best.pt \\
        --mask2022 data/raw/BT2022.gpkg \\
        --out outputs/construction_prob_2024.tif \\
        --out-gpkg outputs/construction_sites_2024.gpkg \\
        --threshold 0.5

The script:
  1. Downloads 2022 and 2024 PDOK tiles for the bounding box.
  2. Rasterises the 2022 construction polygons as a binary mask.
  3. Tiles the images into 1024×1024 patches (with overlap).
  4. Runs ConstructionChangeDetector on each tile pair.
  5. Stitches probability maps with Gaussian blending.
  6. Writes GeoTIFF and/or GeoPackage outputs.
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
    p.add_argument("--checkpoint",  default=None,
                   help="Trained model checkpoint. Omit for zero-shot mode.")
    p.add_argument("--zero-shot",   action="store_true",
                   help="Skip trained scorer — use raw SAM2 cosine similarity only.")
    p.add_argument("--mask2022",    required=True,
                   help="Path to 2022 construction polygons (.gpkg or .shp).")
    p.add_argument("--out",         default=None, help="Output GeoTIFF path (.tif).")
    p.add_argument("--out-gpkg",    default=None, help="Output GeoPackage path (.gpkg).")
    p.add_argument("--threshold",   type=float, default=0.5,
                   help="Probability threshold for GeoPackage vectorisation (default 0.5).")
    p.add_argument("--config",      default="configs/inference.yaml")
    p.add_argument("--year-a",      type=int, default=2022)
    p.add_argument("--year-b",      type=int, default=2024)
    p.add_argument("--tile-size",   type=int, default=1024)
    p.add_argument("--overlap",     type=int, default=128)
    args = p.parse_args()
    if args.out is None and args.out_gpkg is None:
        p.error("Provide at least one of --out (GeoTIFF) or --out-gpkg (GeoPackage).")
    return args


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


def vectorise_to_gpkg(
    prob_combined: np.ndarray,
    prob_preexisting: np.ndarray,
    prob_new: np.ndarray,
    reference_tif: str,
    output_gpkg: str,
    threshold: float = 0.5,
    crs: str = "EPSG:28992",
) -> int:
    """Threshold probability map → vector polygons → GeoPackage.

    Each connected region above *threshold* becomes one polygon feature with
    attributes: type, prob_combined, prob_preexisting, prob_new, area_m2.

    Returns the number of polygons written.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    binary = (prob_combined >= threshold).astype(np.uint8)

    with rasterio.open(reference_tif) as src:
        transform = src.transform

    records = []
    for geom_json, val in rio_shapes(binary, mask=binary, transform=transform):
        if val == 0:
            continue
        poly = shape(geom_json)
        if poly.is_empty or poly.area < 1.0:   # skip sub-pixel slivers
            continue

        # Sample per-polygon mean probabilities from the rasters
        from rasterio.features import rasterize as rio_rasterize
        from rasterio.transform import from_bounds
        minx, miny, maxx, maxy = poly.bounds
        mini_h = max(1, int((maxy - miny) / abs(transform.e)))
        mini_w = max(1, int((maxx - minx) / transform.a))
        mini_t = from_bounds(minx, miny, maxx, maxy, mini_w, mini_h)
        poly_mask = rio_rasterize(
            [(poly, 1)], out_shape=(mini_h, mini_w),
            transform=mini_t, fill=0, dtype="uint8",
        ).astype(bool)

        def _crop(arr: np.ndarray) -> float:
            row0 = max(0, int((prob_combined.shape[0] - 1) - (maxy - transform.f) / abs(transform.e)))
            col0 = max(0, int((minx - transform.c) / transform.a))
            row1 = min(arr.shape[0], row0 + mini_h)
            col1 = min(arr.shape[1], col0 + mini_w)
            patch = arr[row0:row1, col0:col1]
            m = poly_mask[:patch.shape[0], :patch.shape[1]]
            return float(patch[m].mean()) if m.any() else 0.0

        pc = _crop(prob_combined)
        pp = _crop(prob_preexisting)
        pn = _crop(prob_new)
        site_type = "preexisting" if pp >= pn else "new"

        records.append({
            "geometry":         poly,
            "type":             site_type,
            "prob_combined":    round(pc, 4),
            "prob_preexisting": round(pp, 4),
            "prob_new":         round(pn, 4),
            "area_m2":          round(poly.area, 1),
        })

    if not records:
        logger.warning("No polygons above threshold %.2f — empty GeoPackage.", threshold)
        gdf = gpd.GeoDataFrame(columns=[
            "geometry", "type", "prob_combined",
            "prob_preexisting", "prob_new", "area_m2",
        ], geometry="geometry", crs=crs)
    else:
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    Path(output_gpkg).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_gpkg, driver="GPKG", layer="construction_sites")
    logger.info(
        "GeoPackage saved → %s  (%d polygons, threshold=%.2f)",
        output_gpkg, len(gdf), threshold,
    )
    return len(gdf)


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    bbox = tuple(float(v) for v in args.bbox.split(","))
    assert len(bbox) == 4, "bbox must have 4 values: xmin,ymin,xmax,ymax"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | bbox: %s", device, bbox)

    zero_shot = args.zero_shot or args.checkpoint is None
    if zero_shot:
        logger.info("Zero-shot mode — using raw SAM2 cosine similarity (no trained weights).")
    elif args.checkpoint is None:
        raise ValueError("Provide --checkpoint or use --zero-shot flag.")

    # --- Load model ---
    model = ConstructionChangeDetector(k_prototypes=cfg.model.k_prototypes)
    model.load_sam2(cfg.model.sam2_hf_id)
    if not zero_shot:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_trainable_state_dict(state)
    model.to(device).eval()

    proto_ckpt = torch.load(args.prototypes, map_location=device)
    prototypes = proto_ckpt["prototypes"].to(device)   # [K, 256]

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
                if zero_shot:
                    # ── Zero-shot: direct cosine similarity against prototypes ──
                    enc_a = model.image_encoder(t_a)["vision_features"]  # [1,C,Hf,Wf]
                    enc_b = model.image_encoder(t_b)["vision_features"]

                    # Normalise features and prototypes to unit vectors
                    enc_a_n = F.normalize(enc_a, dim=1)          # [1,C,Hf,Wf]
                    enc_b_n = F.normalize(enc_b, dim=1)
                    proto_n = F.normalize(prototypes, dim=1)      # [K,C]

                    # Appearance: max cosine sim over all K prototypes [1,1,Hf,Wf]
                    # Reshape enc_b: [1,C,Hf,Wf] → [Hf*Wf, C]
                    Hf, Wf = enc_b_n.shape[2:]
                    b_flat = enc_b_n.squeeze(0).permute(1, 2, 0).reshape(-1, enc_b_n.shape[1])  # [N,C]
                    sims   = b_flat @ proto_n.T                   # [N, K]
                    appear = sims.max(dim=1).values.reshape(1, 1, Hf, Wf)  # [1,1,Hf,Wf]
                    appear = (appear.clamp(-1, 1) + 1) / 2       # scale [-1,1]→[0,1]

                    # Change: 1 - mean cosine sim between 2022 and 2024 per pixel
                    a_flat  = enc_a_n.squeeze(0).permute(1, 2, 0).reshape(-1, enc_a_n.shape[1])
                    cos_sim = (b_flat * a_flat).sum(dim=1)        # [N]
                    change  = (1 - cos_sim.clamp(-1, 1)).reshape(1, 1, Hf, Wf) / 2  # [0,1]

                    # Upsample back to tile size
                    appear = F.interpolate(appear, size=(t_b.shape[2], t_b.shape[3]), mode="bilinear", align_corners=False)
                    change = F.interpolate(change, size=(t_b.shape[2], t_b.shape[3]), mode="bilinear", align_corners=False)

                    # Routing via 2022 mask (same logic as trained detector)
                    preexisting = appear * m_a
                    new_sites   = appear * change * (1 - m_a)
                    prob_map    = (preexisting + new_sites).clamp(0, 1)

                    out = {
                        "prob_map":    prob_map,
                        "preexisting": preexisting,
                        "new_sites":   new_sites,
                    }
                else:
                    out = model(t_a, t_b, m_a, prototypes)

            combined_tiles.append(out["prob_map"].squeeze().cpu().numpy())
            preexisting_tiles.append(out["preexisting"].squeeze().cpu().numpy())
            new_tiles.append(out["new_sites"].squeeze().cpu().numpy())

        # --- Stitch ---
        shape = (H, W)
        prob_combined    = stitch_tiles(combined_tiles,    coords, shape)
        prob_preexisting = stitch_tiles(preexisting_tiles, coords, shape)
        prob_new         = stitch_tiles(new_tiles,         coords, shape)

        # --- Save GeoTIFF ---
        if args.out:
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
            logger.info("GeoTIFF saved → %s", args.out)
            logger.info("  QGIS: Layer > Add Raster | Symbology: Singleband pseudocolor, Band 1")

        # --- Save GeoPackage ---
        if args.out_gpkg:
            ref = args.out if args.out else path_b
            n = vectorise_to_gpkg(
                prob_combined, prob_preexisting, prob_new,
                reference_tif=ref,
                output_gpkg=args.out_gpkg,
                threshold=args.threshold,
                crs=cfg.data.crs,
            )
            logger.info("  QGIS: Layer > Add Vector | %d construction site polygons", n)


if __name__ == "__main__":
    main()
