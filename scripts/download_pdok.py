"""Download matched (2022, 2024) tile pairs from PDOK Luchtfoto for all
construction polygons in a GeoPackage / Shapefile.

Usage::

    python scripts/download_pdok.py \\
        --polygons data/raw/construction_2022.gpkg \\
        --years 2022 2024 \\
        --resolution 25cm \\
        --padding 50 \\
        --out data/tiles/

The script reprojects polygons to EPSG:28992 (RD New) if needed, then
downloads two GeoTIFF tiles per polygon (one per year) via WMS and rasterises
the polygon as a binary mask PNG.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd

from data.pdok_downloader import PDOKDownloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download PDOK tile pairs for construction polygons.")
    p.add_argument("--polygons",   required=True, help="Path to GeoPackage / Shapefile with construction polygons.")
    p.add_argument("--years",      nargs=2, type=int, default=[2022, 2024], metavar=("YEAR_A", "YEAR_B"))
    p.add_argument("--resolution", default="25cm", choices=["25cm", "8cm"])
    p.add_argument("--padding",    type=float, default=50.0, help="Metres of context padding around each polygon bbox.")
    p.add_argument("--out",        required=True, help="Output root directory.")
    p.add_argument("--sample-type", default="preexisting", choices=["preexisting", "new", "negative"])
    p.add_argument("--crs",        default="EPSG:28992")
    p.add_argument("--delay",      type=float, default=0.5, help="Seconds between WMS requests (rate limiting).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading polygons from %s …", args.polygons)
    gdf = gpd.read_file(args.polygons)
    logger.info("  %d features loaded.", len(gdf))

    if str(gdf.crs) != args.crs:
        logger.info("  Reprojecting from %s → %s …", gdf.crs, args.crs)
        gdf = gdf.to_crs(args.crs)

    downloader = PDOKDownloader(
        resolution=args.resolution,
        crs=args.crs,
        request_delay=args.delay,
    )

    downloader.download_all_polygon_pairs(
        gdf=gdf,
        years=tuple(args.years),
        output_dir=args.out,
        padding_m=args.padding,
        sample_type=args.sample_type,
    )


if __name__ == "__main__":
    main()
