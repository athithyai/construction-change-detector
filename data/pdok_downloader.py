"""Download matched (year_A, year_B) tile pairs from the PDOK Luchtfoto WMS.

PDOK WMS endpoint:
  https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0

Layers:
  Actueel_ortho25  — 25 cm/px colour orthophoto
  Actueel_ortho8   — 8 cm/px colour orthophoto

The service exposes historical years via the TIME dimension
(e.g. TIME=2022-01-01/2022-12-31).

Usage::

    from data.pdok_downloader import PDOKDownloader
    dl = PDOKDownloader(resolution="25cm")
    dl.download_all_polygon_pairs(gdf, years=(2022, 2024), output_dir="data/tiles")
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import geopandas as gpd
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.features import rasterize as rio_rasterize
from owslib.wms import WebMapService
from shapely.geometry import box
from tqdm import tqdm

logger = logging.getLogger(__name__)

WMS_URL = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
LAYER_MAP: Dict[str, str] = {
    "25cm": "Actueel_ortho25",
    "8cm":  "Actueel_ortho8",
}


class PDOKDownloader:
    """Fetch ortophoto tiles from the PDOK Luchtfoto WMS."""

    def __init__(
        self,
        resolution: str = "25cm",
        crs: str = "EPSG:28992",
        request_delay: float = 0.5,
        tile_px: int = 1024,
    ) -> None:
        if resolution not in LAYER_MAP:
            raise ValueError(f"resolution must be one of {list(LAYER_MAP)}")
        self.layer      = LAYER_MAP[resolution]
        self.crs        = crs
        self.delay      = request_delay
        self.tile_px    = tile_px
        self._wms: Optional[WebMapService] = None

    @property
    def wms(self) -> WebMapService:
        if self._wms is None:
            logger.info("Connecting to PDOK WMS …")
            self._wms = WebMapService(WMS_URL, version="1.3.0")
        return self._wms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_tile(
        self,
        bbox: Tuple[float, float, float, float],
        year: int,
        output_path: str,
    ) -> np.ndarray:
        """Download one tile and save it as a georeferenced GeoTIFF.

        Args:
            bbox: (xmin, ymin, xmax, ymax) in self.crs.
            year: Calendar year to fetch (WMS TIME dimension).
            output_path: Destination file path (.tif).

        Returns:
            HxWx3 uint8 numpy array.
        """
        time_str = f"{year}-01-01/{year}-12-31"
        try:
            response = self.wms.getmap(
                layers=[self.layer],
                srs=self.crs,
                bbox=bbox,
                size=(self.tile_px, self.tile_px),
                format="image/png",
                time=time_str,
                transparent=False,
            )
            img_arr = np.array(Image.open(io.BytesIO(response.read())).convert("RGB"))
        except Exception as exc:
            logger.warning("WMS request failed (%s) — retrying once …", exc)
            time.sleep(2.0)
            response = self.wms.getmap(
                layers=[self.layer],
                srs=self.crs,
                bbox=bbox,
                size=(self.tile_px, self.tile_px),
                format="image/png",
                time=time_str,
                transparent=False,
            )
            img_arr = np.array(Image.open(io.BytesIO(response.read())).convert("RGB"))

        self._save_geotiff(img_arr, bbox, output_path)
        time.sleep(self.delay)
        return img_arr

    def download_all_polygon_pairs(
        self,
        gdf: gpd.GeoDataFrame,
        years: Tuple[int, int],
        output_dir: str,
        padding_m: float = 50.0,
        sample_type: str = "preexisting",
    ) -> None:
        """Download (year_A, year_B) tile pairs for every polygon in *gdf*.

        For each polygon the padded bounding box is used as the tile extent.
        A rasterised binary mask (from the polygon) is saved alongside both tiles.

        Directory layout::

            output_dir/
              <site_id>/
                tile_<year_A>.tif
                tile_<year_B>.tif
                mask_<year_A>.png   ← rasterised polygon
                meta.json

        Args:
            gdf:         GeoDataFrame with construction polygons (in self.crs).
            years:       Tuple (year_A, year_B) — typically (2022, 2024).
            output_dir:  Root output directory.
            padding_m:   Metres of context padding around each polygon bbox.
            sample_type: Label written to meta.json ("preexisting" | "new" | "negative").
        """
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        # Ensure correct CRS
        if gdf.crs is None or str(gdf.crs) != self.crs:
            gdf = gdf.to_crs(self.crs)

        year_a, year_b = years
        skipped = 0

        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Downloading tiles"):
            site_id = str(row.get("identificatie", idx))
            site_dir = out_root / site_id
            site_dir.mkdir(exist_ok=True)

            geom = row.geometry
            if geom is None or geom.is_empty:
                skipped += 1
                continue

            xmin, ymin, xmax, ymax = geom.bounds
            bbox = (
                xmin - padding_m,
                ymin - padding_m,
                xmax + padding_m,
                ymax + padding_m,
            )

            path_a = str(site_dir / f"tile_{year_a}.tif")
            path_b = str(site_dir / f"tile_{year_b}.tif")
            mask_path = str(site_dir / f"mask_{year_a}.png")

            # Download tiles
            try:
                if not Path(path_a).exists():
                    self.download_tile(bbox, year_a, path_a)
                if not Path(path_b).exists():
                    self.download_tile(bbox, year_b, path_b)
            except Exception as exc:
                logger.error("Failed site %s: %s", site_id, exc)
                skipped += 1
                continue

            # Rasterise mask
            if not Path(mask_path).exists():
                self._rasterise_mask(geom, bbox, self.tile_px, mask_path)

            # Write metadata
            meta = {
                "site_id":     site_id,
                "bbox":        list(bbox),
                "crs":         self.crs,
                "year_a":      year_a,
                "year_b":      year_b,
                "sample_type": sample_type,
            }
            with open(site_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

        logger.info(
            "Done. Downloaded %d sites, skipped %d.",
            len(gdf) - skipped, skipped,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_geotiff(
        self,
        img: np.ndarray,
        bbox: Tuple[float, float, float, float],
        output_path: str,
    ) -> None:
        """Save HxWx3 uint8 array as a georeferenced GeoTIFF."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        h, w = img.shape[:2]
        transform = from_bounds(*bbox, width=w, height=h)
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=3,
            dtype="uint8",
            crs=self.crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            for band_idx in range(3):
                dst.write(img[:, :, band_idx], band_idx + 1)

    @staticmethod
    def _rasterise_mask(
        geom,
        bbox: Tuple[float, float, float, float],
        tile_px: int,
        output_path: str,
    ) -> None:
        """Burn polygon into a binary uint8 PNG mask."""
        transform = from_bounds(*bbox, width=tile_px, height=tile_px)
        mask = rio_rasterize(
            [(geom, 1)],
            out_shape=(tile_px, tile_px),
            transform=transform,
            fill=0,
            dtype="uint8",
        )
        Image.fromarray(mask * 255).save(output_path)
