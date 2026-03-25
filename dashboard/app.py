"""SamSpade Dashboard — FastAPI backend.

Single mode: SAM2 auto-segment → per-segment CLIP classification + NDVI + depth.
"""
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import io
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from pyproj import Transformer

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

from data.pdok_downloader import PDOKDownloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Global model state ─────────────────────────────────────────────────────────
SAM2_MODEL     = None
SAM2_MASK_GEN  = None
CLIP_MODEL     = None
CLIP_PROCESSOR = None
CLIP_TEXT_FEATS: Optional[torch.Tensor] = None
DEPTH_MODEL    = None
DEPTH_PROCESSOR = None
DEVICE: Optional[torch.device] = None
MODEL_READY    = False
LAST_GPKG: Optional[Path] = None

CIR_WMS_URL = "https://service.pdok.nl/hwh/luchtfotocir/wms/v1_0"

# ── CLIP semantic labels ────────────────────────────────────────────────────────
CLIP_LABELS = [
    "construction site with cranes and machinery",       # 0
    "building under construction concrete frame",        # 1
    "excavation pit earthwork foundation",               # 2
    "scaffolding on building facade",                    # 3
    "demolition site rubble and debris",                 # 4
    "concrete pour steel structure being built",         # 5
    "road asphalt pavement",                             # 6
    "parking lot with vehicles",                         # 7
    "residential house rooftops",                        # 8
    "office or commercial building roof",                # 9
    "industrial warehouse factory roof",                 # 10
    "green vegetation trees park grass",                 # 11
    "water canal river pond",                            # 12
    "bare soil gravel sand field",                       # 13
]
CONSTRUCTION_IDXS = frozenset(range(6))   # labels 0-5

LABEL_COLORS = [
    "#FF4400",  # 0  construction site
    "#FF8C00",  # 1  building under construction
    "#FFD700",  # 2  excavation
    "#FF1493",  # 3  scaffolding
    "#FF6347",  # 4  demolition
    "#FFA500",  # 5  concrete pour
    "#808080",  # 6  road
    "#B0C4DE",  # 7  parking
    "#90EE90",  # 8  residential
    "#87CEEB",  # 9  office/commercial
    "#DDA0DD",  # 10 industrial
    "#2E8B57",  # 11 vegetation
    "#4169E1",  # 12 water
    "#D2B48C",  # 13 bare soil
]

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SamSpade")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC = ROOT / "static"
STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# ── Model loading ──────────────────────────────────────────────────────────────
def _load_model() -> None:
    global SAM2_MODEL, SAM2_MASK_GEN, CLIP_MODEL, CLIP_PROCESSOR, CLIP_TEXT_FEATS
    global DEPTH_MODEL, DEPTH_PROCESSOR, DEVICE, MODEL_READY
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        DEVICE = device

        # SAM2
        logger.info("Loading SAM2 Hiera-L on %s …", device)
        from sam2.build_sam import build_sam2_hf
        SAM2_MODEL = build_sam2_hf("facebook/sam2.1-hiera-large", device=device)
        SAM2_MODEL.eval()

        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        SAM2_MASK_GEN = SAM2AutomaticMaskGenerator(
            model=SAM2_MODEL,
            points_per_side=32,
            pred_iou_thresh=0.70,
            stability_score_thresh=0.80,
            min_mask_region_area=100,
        )
        logger.info("SAM2AutomaticMaskGenerator ready (32 pts/side).")

        # CLIP
        logger.info("Loading CLIP ViT-L/14 …")
        from transformers import CLIPProcessor, CLIPModel
        CLIP_MODEL     = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        CLIP_MODEL.eval()
        with torch.no_grad():
            txt_in = CLIP_PROCESSOR(text=CLIP_LABELS, return_tensors="pt", padding=True).to(device)
            txt_out = CLIP_MODEL.text_model(input_ids=txt_in["input_ids"],
                                            attention_mask=txt_in["attention_mask"])
            CLIP_TEXT_FEATS = F.normalize(CLIP_MODEL.text_projection(txt_out.pooler_output), dim=-1)
        logger.info("CLIP ready — %d label embeddings.", len(CLIP_LABELS))

        # Depth Anything V2 Small
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            logger.info("Loading Depth Anything V2 Small …")
            DEPTH_PROCESSOR = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            DEPTH_MODEL     = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf").to(device)
            DEPTH_MODEL.eval()
            logger.info("Depth Anything V2 ready.")
        except Exception as exc:
            logger.warning("Depth Anything V2 not loaded: %s", exc)

        MODEL_READY = True
        logger.info("All models ready ✓  device=%s", device)
    except Exception as exc:
        logger.error("Model load failed: %s", exc)
        raise


@app.on_event("startup")
async def startup() -> None:
    loop = asyncio.get_event_loop()
    loop.run_in_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1), _load_model)


# ── Schema ─────────────────────────────────────────────────────────────────────
class AutoSegRequest(BaseModel):
    bbox_wgs84: list[float]   # [west, south, east, north]
    year: int = 2024


# ── Geo helpers ────────────────────────────────────────────────────────────────
def _bbox_rd_from_wgs84(bbox_wgs84):
    t = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    w, s = t.transform(bbox_wgs84[0], bbox_wgs84[1])
    e, n = t.transform(bbox_wgs84[2], bbox_wgs84[3])
    return (w, s, e, n)


def _to_wgs84_bounds(bbox_rd):
    t2 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    west,  south = t2.transform(bbox_rd[0], bbox_rd[1])
    east,  north = t2.transform(bbox_rd[2], bbox_rd[3])
    return west, south, east, north


def _enforce_min_extent(bbox_rd, min_half: float = 300.0):
    xmin, ymin, xmax, ymax = bbox_rd
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half   = max((xmax - xmin) / 2, (ymax - ymin) / 2, min_half)
    return (cx - half, cy - half, cx + half, cy + half)


def _fetch_tile(dl_8cm, dl_25cm, bbox_rd, year: int, path: str):
    r = dl_8cm.download_tile(bbox_rd, year, path)
    if r is not None:
        return r
    logger.warning("8 cm blank for year=%d — trying 25 cm…", year)
    path25 = path.replace(".tif", "_25cm.tif")
    r = dl_25cm.download_tile(bbox_rd, year, path25)
    if r is not None:
        import shutil
        shutil.move(path25, path)
        return r
    return None


def _load_geotiff(path: str):
    """Open GeoTIFF, return (HxWx3 uint8, crs, transform)."""
    import rasterio
    with rasterio.open(path) as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0)
        return img.copy(), src.crs, src.transform


# ── CIR / NDVI ─────────────────────────────────────────────────────────────────
def _fetch_cir_tile(bbox_rd: tuple, year: int) -> Optional[np.ndarray]:
    import requests as _req
    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": f"{year}_ortho25", "STYLES": "",
        "CRS": "EPSG:28992",
        "BBOX": f"{bbox_rd[0]},{bbox_rd[1]},{bbox_rd[2]},{bbox_rd[3]}",
        "WIDTH": "512", "HEIGHT": "512",
        "FORMAT": "image/png",
    }
    try:
        r = _req.get(CIR_WMS_URL, params=params, timeout=15)
        ct = r.headers.get("Content-Type", "")
        if "xml" in ct or not r.content:
            return None
        arr = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
        if (arr == 255).all(axis=-1).mean() > 0.95:
            return None
        return arr
    except Exception as exc:
        logger.warning("CIR fetch failed: %s", exc)
        return None


def _compute_ndvi(cir_np: np.ndarray, target_hw: tuple) -> np.ndarray:
    """CIR → NDVI in [-1,1].  R band = NIR, G band = Red."""
    import cv2
    nir  = cir_np[:, :, 0].astype(np.float32)
    red  = cir_np[:, :, 1].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)
    H, W = target_hw
    return cv2.resize(ndvi, (W, H), interpolation=cv2.INTER_LINEAR)


def _ndvi_to_overlay(ndvi: np.ndarray) -> str:
    """Colorize NDVI: red (bare/construction) → yellow (transition) → green (vegetation)."""
    import cv2
    norm = np.clip((ndvi + 1) / 2, 0, 1)   # [-1,1] → [0,1]
    cm   = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(cm, cv2.COLORMAP_RdYlGn)  # BGR
    rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    alpha = np.full((*rgb.shape[:2], 1), 200, dtype=np.uint8)
    rgba  = np.concatenate([rgb, alpha], axis=-1)
    buf   = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── Depth ──────────────────────────────────────────────────────────────────────
def _compute_depth(img_np: np.ndarray) -> Optional[np.ndarray]:
    """Depth Anything V2 → HxW float32 depth map, normalized [0,1]."""
    if DEPTH_MODEL is None:
        return None
    import cv2
    H, W = img_np.shape[:2]
    try:
        with torch.no_grad():
            inp   = DEPTH_PROCESSOR(images=Image.fromarray(img_np), return_tensors="pt").to(DEVICE)
            depth = DEPTH_MODEL(**inp).predicted_depth.squeeze().cpu().numpy().astype(np.float32)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        mn, mx = depth.min(), depth.max()
        return (depth - mn) / (mx - mn + 1e-6)
    except Exception as exc:
        logger.warning("Depth failed: %s", exc)
        return None


def _depth_to_overlay(depth: np.ndarray) -> str:
    """Colorize depth: near=warm, far=cool."""
    import cv2
    cm    = (depth * 255).astype(np.uint8)
    colored = cv2.applyColorMap(cm, cv2.COLORMAP_PLASMA)
    rgb   = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    alpha = np.full((*rgb.shape[:2], 1), 200, dtype=np.uint8)
    rgba  = np.concatenate([rgb, alpha], axis=-1)
    buf   = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── CLIP per-segment classification ────────────────────────────────────────────
def _clip_classify_masks(img_np: np.ndarray, masks_data: list) -> list:
    """Batch-classify each SAM mask crop with CLIP. Returns list of dicts."""
    if CLIP_MODEL is None or not masks_data:
        return [{"top_idx": [0], "top_scores": [0.0]}] * len(masks_data)
    pad = 8
    H, W = img_np.shape[:2]
    crops = []
    for m in masks_data:
        x, y, bw, bh = [int(v) for v in m["bbox"]]
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + bw + pad), min(H, y + bh + pad)
        crop = img_np[y1:y2, x1:x2].copy()
        seg_crop = m["segmentation"][y1:y2, x1:x2]
        crop[~seg_crop] = 128   # gray outside mask
        crops.append(Image.fromarray(crop))
    results, bs = [], 32
    with torch.no_grad():
        for i in range(0, len(crops), bs):
            batch = crops[i: i + bs]
            inp   = CLIP_PROCESSOR(images=batch, return_tensors="pt", padding=True).to(DEVICE)
            v_out = CLIP_MODEL.vision_model(pixel_values=inp["pixel_values"])
            feats = F.normalize(CLIP_MODEL.visual_projection(v_out.pooler_output), dim=-1)
            sims  = (feats @ CLIP_TEXT_FEATS.T).cpu()
            for j in range(len(batch)):
                top = sims[j].topk(3)
                results.append({
                    "top_idx":    top.indices.tolist(),
                    "top_scores": top.values.tolist(),
                })
    return results


# ── Vectorization ──────────────────────────────────────────────────────────────
def _masks_to_gdf(masks_sorted: list, img_shape: tuple, tile_crs, tile_transform):
    """SAM2 masks (sorted descending) → GeoDataFrame in WGS84."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    H, W = img_shape[:2]
    labeled = np.zeros((H, W), dtype=np.int32)
    for idx, m in enumerate(masks_sorted, start=1):
        labeled[m["segmentation"]] = idx

    polys_by_id: dict = {}
    for geom_j, val in rio_shapes(labeled, mask=(labeled > 0).astype(np.uint8),
                                   transform=tile_transform):
        v = int(val)
        p = shapely_shape(geom_j)
        if not p.is_empty and p.area > 0.5:
            polys_by_id.setdefault(v, []).append(p)

    rows = [{"mask_id": v, "geometry": unary_union(polys)}
            for v, polys in polys_by_id.items()]
    if not rows:
        return gpd.GeoDataFrame(columns=["mask_id", "geometry"], crs=tile_crs)
    gdf = gpd.GeoDataFrame(rows, crs=tile_crs)
    return gdf.to_crs("EPSG:4326")


# ── Segment signal extraction ──────────────────────────────────────────────────
def _mask_mean(signal: np.ndarray, seg: np.ndarray, target_hw: tuple) -> float:
    """Mean of a signal map within a SAM mask, resized to target_hw if needed."""
    import cv2
    H, W = target_hw
    sh, sw = signal.shape[:2]
    if (sh, sw) != (H, W):
        signal = cv2.resize(signal, (W, H), interpolation=cv2.INTER_LINEAR)
    seg_r = seg if seg.shape == (H, W) else cv2.resize(
        seg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    vals = signal[seg_r]
    return float(vals.mean()) if len(vals) > 0 else 0.0


def _depth_variance(depth: np.ndarray, seg: np.ndarray, target_hw: tuple) -> float:
    """Local depth variance within a segment (construction = rough = high variance)."""
    import cv2
    H, W = target_hw
    sh, sw = depth.shape[:2]
    if (sh, sw) != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    seg_r = seg if seg.shape == (H, W) else cv2.resize(
        seg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    vals = depth[seg_r]
    return float(np.var(vals)) if len(vals) > 1 else 0.0


# ── Main: auto-segment ─────────────────────────────────────────────────────────
def _run_auto_segment(bbox_rd: tuple, year: int) -> dict:
    """
    1. Download RGB tile (PDOK) + CIR tile (PDOK CIR)
    2. SAM2 auto-segment → full pixel coverage
    3. Depth Anything V2 → depth map
    4. NDVI from CIR
    5. For each segment:
       - CLIP label
       - Mean NDVI  (negative = bare soil / construction)
       - Depth variance  (high = rough terrain)
    Returns segments GeoJSON + NDVI overlay PNG + depth overlay PNG
    """
    import geopandas as gpd
    from shapely.geometry import box as shapely_box

    global LAST_GPKG

    original_bbox_rd = bbox_rd
    bbox_rd = _enforce_min_extent(bbox_rd)

    with tempfile.TemporaryDirectory() as tmp:
        dl_8  = PDOKDownloader(resolution="8cm",  crs="EPSG:28992", request_delay=0)
        dl_25 = PDOKDownloader(resolution="25cm", crs="EPSG:28992", request_delay=0)
        tile_path = str(Path(tmp) / f"tile_{year}.tif")

        if _fetch_tile(dl_8, dl_25, bbox_rd, year, tile_path) is None:
            raise HTTPException(404, f"No PDOK coverage for {year} at this location.")

        img, tile_crs, tile_transform = _load_geotiff(tile_path)
        H, W = img.shape[:2]
        logger.info("Tile loaded: %dx%d", H, W)

        # ── NDVI ──────────────────────────────────────────────────────────────
        ndvi_map: Optional[np.ndarray] = None
        ndvi_overlay: Optional[str]    = None
        cir = _fetch_cir_tile(bbox_rd, year)
        if cir is not None:
            ndvi_map     = _compute_ndvi(cir, (H, W))
            ndvi_overlay = _ndvi_to_overlay(ndvi_map)
            logger.info("NDVI ready.")
        else:
            logger.warning("CIR tile unavailable — NDVI skipped.")

        # ── Depth ─────────────────────────────────────────────────────────────
        depth_map: Optional[np.ndarray] = None
        depth_overlay: Optional[str]    = None
        depth_map = _compute_depth(img)
        if depth_map is not None:
            depth_overlay = _depth_to_overlay(depth_map)
            logger.info("Depth map ready.")

        # ── SAM2 auto-segment ─────────────────────────────────────────────────
        logger.info("SAM2 auto-segment %dx%d …", H, W)
        masks_data   = SAM2_MASK_GEN.generate(img)
        logger.info("  → %d raw masks", len(masks_data))
        masks_sorted = sorted(masks_data, key=lambda m: m["area"], reverse=True)

        # ── Vectorize ─────────────────────────────────────────────────────────
        gdf_wgs = _masks_to_gdf(masks_sorted, img.shape, tile_crs, tile_transform)
        logger.info("  → %d vector features", len(gdf_wgs))

        # Clip to user's drawn area (remove PDOK-expanded padding)
        t2 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
        ox1, oy1 = t2.transform(original_bbox_rd[0], original_bbox_rd[1])
        ox2, oy2 = t2.transform(original_bbox_rd[2], original_bbox_rd[3])
        clip_box = shapely_box(ox1, oy1, ox2, oy2)
        gdf_wgs  = gdf_wgs[gdf_wgs.geometry.intersects(clip_box)].copy()
        gdf_wgs["geometry"] = gdf_wgs.geometry.intersection(clip_box)
        gdf_wgs  = gdf_wgs[~gdf_wgs.geometry.is_empty].reset_index(drop=True)
        logger.info("  → %d features after clip", len(gdf_wgs))

        # ── CLIP classify ─────────────────────────────────────────────────────
        logger.info("  → CLIP classifying %d segments …", len(masks_sorted))
        clip_res = _clip_classify_masks(img, masks_sorted)

        # ── Build GeoJSON ─────────────────────────────────────────────────────
        records, features_json = [], []
        label_counts: dict = {}

        for _, row in gdf_wgs.iterrows():
            mask_idx = int(row["mask_id"]) - 1
            if mask_idx < 0 or mask_idx >= len(masks_sorted):
                continue
            mdata = masks_sorted[mask_idx]
            cr    = clip_res[mask_idx] if mask_idx < len(clip_res) else {"top_idx": [0], "top_scores": [0.0]}
            seg   = mdata["segmentation"]   # HxW bool at tile resolution

            label_idx = cr["top_idx"][0]
            label     = CLIP_LABELS[label_idx]
            clip_score = float(cr["top_scores"][0])
            is_constr  = label_idx in CONSTRUCTION_IDXS
            color      = LABEL_COLORS[label_idx] if label_idx < len(LABEL_COLORS) else "#aaaaaa"
            top3       = [{"label": CLIP_LABELS[i], "score": round(float(s), 3)}
                          for i, s in zip(cr["top_idx"], cr["top_scores"])]

            # Per-segment NDVI and depth
            mean_ndvi   = round(_mask_mean(ndvi_map,  seg, (H, W)), 3) if ndvi_map  is not None else None
            depth_var   = round(_depth_variance(depth_map, seg, (H, W)) * 1000, 3) if depth_map is not None else None

            records.append({
                "geometry":        row.geometry,
                "label":           label,
                "clip_score":      round(clip_score, 3),
                "is_construction": is_constr,
                "area_m2":         round(float(mdata["area"]), 1),
                "stability_score": round(float(mdata.get("stability_score", 0)), 3),
                "predicted_iou":   round(float(mdata.get("predicted_iou",   0)), 3),
                "mean_ndvi":       mean_ndvi,
                "depth_variance":  depth_var,
                "_color":          color,
                "_top3":           top3,
            })
            label_counts[label] = label_counts.get(label, 0) + 1

            features_json.append({
                "type": "Feature",
                "geometry": row.geometry.__geo_interface__,
                "properties": {
                    "label":           label,
                    "clip_score":      round(clip_score, 3),
                    "color":           color,
                    "is_construction": is_constr,
                    "top3":            top3,
                    "area_m2":         round(float(mdata["area"]), 1),
                    "stability_score": round(float(mdata.get("stability_score", 0)), 3),
                    "predicted_iou":   round(float(mdata.get("predicted_iou",   0)), 3),
                    "mean_ndvi":       mean_ndvi,
                    "depth_variance":  depth_var,
                },
            })

        if not records:
            west, south, east, north = _to_wgs84_bounds(original_bbox_rd)
            return {
                "geojson":      {"type": "FeatureCollection", "features": []},
                "bounds":       [[south, west], [north, east]],
                "ndvi_overlay": ndvi_overlay,
                "depth_overlay": depth_overlay,
                "stats":        {"total": 0, "construction": 0, "labels": {}},
            }

        # Construction segments on top
        features_json.sort(key=lambda f: (0 if f["properties"]["is_construction"] else 1,
                                          -f["properties"]["area_m2"]))

        # Save GeoPackage
        gdf_out  = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        gpkg_out = ROOT.parent / "outputs" / f"segments_{year}.gpkg"
        gpkg_out.parent.mkdir(exist_ok=True)
        gpkg_cols = [c for c in gdf_out.columns if not c.startswith("_")]
        gdf_out[gpkg_cols].to_file(str(gpkg_out), driver="GPKG", layer="segments")
        LAST_GPKG = gpkg_out
        logger.info("GeoPackage saved → %s", gpkg_out)

        n_constr = sum(1 for f in features_json if f["properties"]["is_construction"])
        west, south, east, north = _to_wgs84_bounds(original_bbox_rd)
        return {
            "geojson":       {"type": "FeatureCollection", "features": features_json},
            "bounds":        [[south, west], [north, east]],
            "ndvi_overlay":  ndvi_overlay,
            "depth_overlay": depth_overlay,
            "stats": {
                "total":        len(features_json),
                "construction": n_constr,
                "labels":       label_counts,
            },
        }


# ── API endpoints ──────────────────────────────────────────────────────────────
@app.post("/api/auto-segment")
async def auto_segment(req: AutoSegRequest) -> dict:
    if not MODEL_READY:
        raise HTTPException(503, "Models loading — check /api/health.")
    bbox_rd = _bbox_rd_from_wgs84(req.bbox_wgs84)
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=1),
            _run_auto_segment, bbox_rd, req.year,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Auto-segment error")
        raise HTTPException(500, str(exc))


@app.get("/api/bt2022")
async def bt2022_geojson() -> dict:
    """Return BT2022 construction polygons as GeoJSON (WGS84, simplified)."""
    gpkg = ROOT.parent / "data/raw/BT2022.gpkg"
    if not gpkg.exists():
        return {"type": "FeatureCollection", "features": []}
    try:
        import geopandas as gpd
        gdf = gpd.read_file(str(gpkg))
        if str(gdf.crs) != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        # Simplify geometry for web transfer (1 m tolerance in degrees ≈ 0.00001)
        gdf["geometry"] = gdf.geometry.simplify(0.00005, preserve_topology=True)
        gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
        # Keep only essential columns
        keep = ["geometry"]
        for col in ["identificatie", "naam", "typegebouw", "oppervlakte"]:
            if col in gdf.columns:
                keep.append(col)
        import json
        return json.loads(gdf[keep].to_json())
    except Exception as exc:
        logger.warning("BT2022 load failed: %s", exc)
        return {"type": "FeatureCollection", "features": []}


@app.get("/api/health")
async def health() -> dict:
    return {"ready": MODEL_READY, "device": str(DEVICE) if DEVICE else "loading"}


@app.get("/api/download/results.gpkg")
async def download_gpkg() -> FileResponse:
    if LAST_GPKG is None or not LAST_GPKG.exists():
        raise HTTPException(404, "No results yet — run a segmentation first.")
    return FileResponse(str(LAST_GPKG), media_type="application/geopackage+sqlite3",
                        filename="segments.gpkg")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")
