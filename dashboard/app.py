"""SamSpade Dashboard — FastAPI backend."""
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import io
import json
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
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel
from pyproj import Transformer

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

from data.pdok_downloader import PDOKDownloader
from data.transforms import get_inference_transforms
from models.detector import ConstructionChangeDetector
from models.feature_utils import load_geotiff_rgb, stitch_tiles, tile_image_for_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Global model state ─────────────────────────────────────────────────────────
MODEL: Optional[ConstructionChangeDetector] = None
PROTOTYPES: Optional[torch.Tensor] = None
DEVICE: Optional[torch.device] = None
CFG = None
MODEL_READY = False
LAST_GPKG: Optional[Path] = None
SAM2_MASK_GEN = None
CLIP_MODEL = None
CLIP_PROCESSOR = None
CLIP_TEXT_FEATS: Optional[torch.Tensor] = None
DEPTH_MODEL = None
DEPTH_PROCESSOR = None
ONE_SHOT_ENCODING: Optional[torch.Tensor] = None   # [C] mean-pooled SAM2 features
ONE_SHOT_CLIP: Optional[torch.Tensor] = None        # [512] CLIP image features
ONE_SHOT_PREVIEW: Optional[str] = None              # base64 thumbnail

CIR_WMS_URL = "https://service.pdok.nl/hwh/luchtfotocir/wms/v1_0"

# ── CLIP semantic labels for aerial construction imagery ────────────────────────
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
CONSTRUCTION_IDXS = frozenset(range(6))   # labels 0-5 are construction-related
LABEL_COLORS_LIST = [
    "#FF4400",  # 0  construction site — vivid orange-red
    "#FF8C00",  # 1  building under construction — dark orange
    "#FFD700",  # 2  excavation — gold
    "#FF1493",  # 3  scaffolding — deep pink
    "#FF6347",  # 4  demolition — tomato
    "#FFA500",  # 5  concrete pour — orange
    "#A0A0A0",  # 6  road — medium gray
    "#B0C4DE",  # 7  parking — light steel blue
    "#90EE90",  # 8  residential — light green
    "#87CEEB",  # 9  office/commercial — sky blue
    "#DDA0DD",  # 10 industrial — plum
    "#2E8B57",  # 11 vegetation — sea green
    "#4169E1",  # 12 water — royal blue
    "#D2B48C",  # 13 bare soil — tan
]

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SamSpade")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC = ROOT / "static"
STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# ── Model loading ──────────────────────────────────────────────────────────────
def _load_model() -> None:
    global MODEL, PROTOTYPES, DEVICE, CFG, MODEL_READY
    global SAM2_MASK_GEN, CLIP_MODEL, CLIP_PROCESSOR, CLIP_TEXT_FEATS
    global DEPTH_MODEL, DEPTH_PROCESSOR
    try:
        cfg = OmegaConf.load(str(ROOT.parent / "configs/base.yaml"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading SAM2 (%s) on %s …", cfg.model.sam2_hf_id, device)

        model = ConstructionChangeDetector(k_prototypes=cfg.model.k_prototypes)
        model.load_sam2(cfg.model.sam2_hf_id)
        model.to(device).eval()

        proto_ckpt = torch.load(str(ROOT.parent / "checkpoints/prototypes.pt"), map_location=device)
        MODEL      = model
        PROTOTYPES = proto_ckpt["prototypes"].to(device)
        DEVICE     = device
        CFG        = cfg

        # SAM2 AutomaticMaskGenerator — 32 pts/side = 1024 prompt points, full pixel coverage
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        SAM2_MASK_GEN = SAM2AutomaticMaskGenerator(
            model=model.sam2_model,
            points_per_side=32,
            pred_iou_thresh=0.70,
            stability_score_thresh=0.80,
            min_mask_region_area=100,
        )
        logger.info("SAM2AutomaticMaskGenerator ready (32 pts/side).")

        # CLIP ViT-L/14 for per-segment semantic classification
        logger.info("Loading CLIP ViT-L/14 …")
        from transformers import CLIPProcessor, CLIPModel
        clip_id = "openai/clip-vit-large-patch14"
        CLIP_MODEL     = CLIPModel.from_pretrained(clip_id).to(device)
        CLIP_PROCESSOR = CLIPProcessor.from_pretrained(clip_id)
        CLIP_MODEL.eval()
        with torch.no_grad():
            txt_in = CLIP_PROCESSOR(text=CLIP_LABELS, return_tensors="pt", padding=True).to(device)
            txt_out = CLIP_MODEL.text_model(input_ids=txt_in["input_ids"],
                                            attention_mask=txt_in["attention_mask"])
            CLIP_TEXT_FEATS = F.normalize(CLIP_MODEL.text_projection(txt_out.pooler_output), dim=-1)
        logger.info("CLIP ready — %d label embeddings.", len(CLIP_LABELS))

        # Depth Anything V2 Small — optional, used as terrain-variance signal
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            logger.info("Loading Depth Anything V2 Small …")
            DEPTH_PROCESSOR = AutoImageProcessor.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf")
            DEPTH_MODEL = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf").to(device)
            DEPTH_MODEL.eval()
            logger.info("Depth Anything V2 ready.")
        except Exception as exc:
            logger.warning("Depth Anything V2 not loaded (skipped): %s", exc)

        MODEL_READY = True
        logger.info("All models ready ✓  device=%s", device)
    except Exception as exc:
        logger.error("Model load failed: %s", exc)
        raise


@app.on_event("startup")
async def startup() -> None:
    loop = asyncio.get_event_loop()
    loop.run_in_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1), _load_model)


# ── Schemas ────────────────────────────────────────────────────────────────────
class DetectRequest(BaseModel):
    bbox_wgs84: list[float]   # [west, south, east, north]
    threshold: float = 0.30
    year_a: int = 2022
    year_b: int = 2024



class AutoSegRequest(BaseModel):
    bbox_wgs84: list[float]    # [west, south, east, north]
    year: int = 2024


class EncodeReferenceRequest(BaseModel):
    bbox_wgs84: list[float]
    year: int = 2024


class OneShotSearchRequest(BaseModel):
    bbox_wgs84: list[float]
    year: int = 2024
    threshold: float = 0.45


# ── Helpers ────────────────────────────────────────────────────────────────────
def _prob_to_rgba(prob: np.ndarray) -> bytes:
    h, w = prob.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 255
    rgba[:, :, 1] = ((1 - prob) * 160).clip(0, 160).astype(np.uint8)
    rgba[:, :, 3] = (prob * 220).clip(0, 220).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _enforce_min_extent(bbox_rd, min_half: float = 300.0):
    """Expand bbox so each side is at least min_half*2 metres (PDOK minimum ~600 m)."""
    xmin, ymin, xmax, ymax = bbox_rd
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half_x = max((xmax - xmin) / 2, min_half)
    half_y = max((ymax - ymin) / 2, min_half)
    half   = max(half_x, half_y)
    return (cx - half, cy - half, cx + half, cy + half)


def _fetch_tile(dl_8cm, dl_25cm, bbox_rd, year: int, path: str):
    """Try 8 cm first, fall back to 25 cm.  Returns None only if both fail."""
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


def _download_pair(bbox_rd, year_a, year_b, tmp):
    """Download two tiles. Returns (expanded_bbox_rd, path_a, path_b) or raises HTTPException."""
    bbox_rd = _enforce_min_extent(bbox_rd)
    dl_8  = PDOKDownloader(resolution="8cm",  crs="EPSG:28992", request_delay=0)
    dl_25 = PDOKDownloader(resolution="25cm", crs="EPSG:28992", request_delay=0)
    path_a = str(Path(tmp) / f"tile_{year_a}.tif")
    path_b = str(Path(tmp) / f"tile_{year_b}.tif")
    logger.info("Fetching %d tile  bbox_rd=%s …", year_a, bbox_rd)
    r_a = _fetch_tile(dl_8, dl_25, bbox_rd, year_a, path_a)
    logger.info("Fetching %d tile  bbox_rd=%s …", year_b, bbox_rd)
    r_b = _fetch_tile(dl_8, dl_25, bbox_rd, year_b, path_b)
    if r_a is None or not Path(path_a).exists():
        raise HTTPException(404, f"No PDOK coverage for year {year_a} at this location. bbox_rd={bbox_rd}")
    if r_b is None or not Path(path_b).exists():
        raise HTTPException(404, f"No PDOK coverage for year {year_b} at this location. bbox_rd={bbox_rd}")
    return bbox_rd, path_a, path_b


def _bbox_rd_from_wgs84(bbox_wgs84):
    t = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    w, s = t.transform(bbox_wgs84[0], bbox_wgs84[1])
    e, n = t.transform(bbox_wgs84[2], bbox_wgs84[3])
    return (w, s, e, n)


def _to_wgs84_bounds(bbox_rd):
    t2 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    west, south = t2.transform(bbox_rd[0], bbox_rd[1])
    east, north = t2.transform(bbox_rd[2], bbox_rd[3])
    return west, south, east, north


# ── Depth + NDVI helpers ───────────────────────────────────────────────────────
def _fetch_cir_tile(bbox_rd: tuple, year: int) -> "Optional[np.ndarray]":
    """Download CIR tile from PDOK CIR WMS. Returns HxWx3 uint8 or None."""
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
        logger.warning("CIR tile fetch failed: %s", exc)
        return None


def _compute_ndvi_signal(cir_np: np.ndarray, target_shape: tuple) -> np.ndarray:
    """NDVI from CIR image (R=NIR, G=Red). Low NDVI = bare soil = construction signal."""
    import cv2
    nir  = cir_np[:, :, 0].astype(np.float32)
    red  = cir_np[:, :, 1].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)  # [-1, 1], low = bare soil
    # Construction signal: NDVI < -0.1 → 1.0,  NDVI > 0.3 → 0.0
    sig = np.clip((0.1 - ndvi) / 0.4, 0.0, 1.0).astype(np.float32)
    H, W = target_shape[:2]
    return cv2.resize(sig, (W, H), interpolation=cv2.INTER_LINEAR)


def _compute_depth_signal(img_np: np.ndarray) -> "Optional[np.ndarray]":
    """Depth Anything V2 → local depth variance as construction signal."""
    if DEPTH_MODEL is None:
        return None
    import cv2
    H, W = img_np.shape[:2]
    pil = Image.fromarray(img_np)
    try:
        with torch.no_grad():
            inp  = DEPTH_PROCESSOR(images=pil, return_tensors="pt").to(DEVICE)
            out  = DEPTH_MODEL(**inp)
            depth = out.predicted_depth.squeeze().cpu().numpy().astype(np.float32)
        depth_r = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        # Local variance with 31×31 kernel
        k = np.ones((31, 31), np.float32) / 961
        mean_  = cv2.filter2D(depth_r, -1, k)
        sq_m   = cv2.filter2D(depth_r ** 2, -1, k)
        var    = np.maximum(sq_m - mean_ ** 2, 0.0)
        vmin, vmax = var.min(), var.max()
        if vmax - vmin < 1e-6:
            return np.zeros((H, W), dtype=np.float32)
        return ((var - vmin) / (vmax - vmin)).astype(np.float32)
    except Exception as exc:
        logger.warning("Depth signal failed: %s", exc)
        return None


# ── CLIP helpers ───────────────────────────────────────────────────────────────
def _clip_spatial_score(img_np: np.ndarray, grid: int = 4) -> np.ndarray:
    """Divide image into grid×grid patches, run CLIP, return (H,W) construction score map."""
    if CLIP_MODEL is None:
        return np.zeros(img_np.shape[:2], dtype=np.float32)
    H, W = img_np.shape[:2]
    ph, pw = H // grid, W // grid
    crops = []
    for i in range(grid):
        for j in range(grid):
            crops.append(Image.fromarray(img_np[i*ph:(i+1)*ph, j*pw:(j+1)*pw]))
    with torch.no_grad():
        inp    = CLIP_PROCESSOR(images=crops, return_tensors="pt", padding=True).to(DEVICE)
        v_out  = CLIP_MODEL.vision_model(pixel_values=inp["pixel_values"])
        feats  = F.normalize(CLIP_MODEL.visual_projection(v_out.pooler_output), dim=-1)
        sims   = feats @ CLIP_TEXT_FEATS.T                                   # [G², n_labels]
        constr_scores = sims[:, list(CONSTRUCTION_IDXS)].max(dim=-1).values.cpu().numpy()
    mn, mx = constr_scores.min(), constr_scores.max()
    constr_scores = (constr_scores - mn) / (mx - mn + 1e-6)
    import cv2
    return cv2.resize(
        constr_scores.reshape(grid, grid).astype(np.float32), (W, H),
        interpolation=cv2.INTER_LINEAR,
    )


def _clip_classify_masks(img_np: np.ndarray, masks_data: list) -> list:
    """Batch-classify each SAM mask crop with CLIP. Returns list of dicts."""
    if CLIP_MODEL is None or not masks_data:
        return [{"top_idx": [0], "top_scores": [0.0]}] * len(masks_data)
    pad = 8
    H, W = img_np.shape[:2]
    crops = []
    for m in masks_data:
        x, y, bw, bh = [int(v) for v in m["bbox"]]  # XYWH from SAM2
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + bw + pad), min(H, y + bh + pad)
        crop = img_np[y1:y2, x1:x2].copy()
        seg_crop = m["segmentation"][y1:y2, x1:x2]
        crop[~seg_crop] = 128   # gray background outside mask
        crops.append(Image.fromarray(crop))
    results, bs = [], 32
    with torch.no_grad():
        for i in range(0, len(crops), bs):
            batch  = crops[i : i + bs]
            inp    = CLIP_PROCESSOR(images=batch, return_tensors="pt", padding=True).to(DEVICE)
            v_out  = CLIP_MODEL.vision_model(pixel_values=inp["pixel_values"])
            feats  = F.normalize(CLIP_MODEL.visual_projection(v_out.pooler_output), dim=-1)
            sims   = (feats @ CLIP_TEXT_FEATS.T).cpu()
            for j in range(len(batch)):
                top = sims[j].topk(3)
                results.append({
                    "top_idx":    top.indices.tolist(),
                    "top_scores": top.values.tolist(),
                })
    return results


# ── SAM2 segment-level feature helper ──────────────────────────────────────────
def _segment_mean_features(feats_n: torch.Tensor, mask_np: np.ndarray) -> torch.Tensor:
    """Extract L2-normalized mean SAM2 features for one segment mask.

    Args:
        feats_n: [1, C, Hf, Wf]  — already L2-normalized feature map
        mask_np: [H, W] bool      — full-resolution segment mask
    Returns:
        [C] normalized mean feature vector
    """
    Hf, Wf = feats_n.shape[2:]
    m_t     = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    m_small = F.interpolate(m_t, size=(Hf, Wf), mode="nearest").squeeze()  # [Hf, Wf]
    f_flat  = feats_n.squeeze(0).permute(1, 2, 0).reshape(-1, feats_n.shape[1])  # [Hf*Wf, C]
    m_flat  = m_small.reshape(-1).bool()
    if m_flat.sum() == 0:
        return F.normalize(f_flat.mean(dim=0), dim=0)
    return F.normalize(f_flat[m_flat].mean(dim=0), dim=0)


# ── samgeo-powered vectorization ───────────────────────────────────────────────
def _samgeo_masks_to_gdf(masks_sorted: list, img_shape: tuple,
                          tile_crs, tile_transform) -> "gpd.GeoDataFrame":
    """
    Convert SAM2 mask list → labeled GeoTIFF → GeoDataFrame in WGS84.
    Uses the same raster_to_vector pipeline as samgeo.common.raster_to_vector
    but correctly handles single-band int32 rasters.
    """
    import rasterio
    import geopandas as gpd
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    H, W = img_shape[:2]
    # Build labeled array: caller must pass masks sorted DESCENDING by area.
    # Large masks are written first; smaller masks overwrite them → every pixel
    # ends up labelled with its *smallest* enclosing mask (full pixel coverage).
    labeled = np.zeros((H, W), dtype=np.int32)
    for idx, m in enumerate(masks_sorted, start=1):
        labeled[m["segmentation"]] = idx

    # Vectorize — samgeo style: shapes per unique label value, then group into one poly per mask
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
    return gdf.to_crs("EPSG:4326")   # samgeo always reprojects to WGS84 for web maps


# ── Auto-segment (samgeo pipeline + SAM2 masks + CLIP labels) ──────────────────
def _run_auto_segment(bbox_rd: tuple, year: int) -> dict:
    """
    SAM2 AutomaticMaskGenerator → full pixel coverage → CLIP per-mask classify → GeoJSON.
    Masks are sorted DESCENDING by area so small masks overwrite large ones in the
    labeled raster — every pixel is assigned to its smallest enclosing segment.
    Results are clipped to the user's originally drawn bbox (not the PDOK-expanded one).
    """
    import geopandas as gpd
    from shapely.geometry import box as shapely_box

    global LAST_GPKG

    original_bbox_rd = bbox_rd          # save before PDOK expansion
    bbox_rd = _enforce_min_extent(bbox_rd)

    with tempfile.TemporaryDirectory() as tmp:
        dl_8  = PDOKDownloader(resolution="8cm",  crs="EPSG:28992", request_delay=0)
        dl_25 = PDOKDownloader(resolution="25cm", crs="EPSG:28992", request_delay=0)
        tile_path = str(Path(tmp) / f"tile_{year}.tif")
        if _fetch_tile(dl_8, dl_25, bbox_rd, year, tile_path) is None:
            raise HTTPException(404, f"No PDOK coverage for {year}. bbox_rd={bbox_rd}")

        img, src = load_geotiff_rgb(tile_path)
        tile_crs       = src.crs
        tile_transform = src.transform
        src.close()
        H, W = img.shape[:2]

        # SAM2 AutoMaskGen — 32 pts/side = 1024 prompts
        logger.info("SAM2 AutoMaskGen on %dx%d …", H, W)
        masks_data = SAM2_MASK_GEN.generate(img)
        logger.info("  → %d raw masks", len(masks_data))

        # Sort DESCENDING by area: large masks first, small masks overwrite → full coverage
        masks_sorted = sorted(masks_data, key=lambda m: m["area"], reverse=True)

        # Vectorize: labeled raster → GDF in WGS84
        gdf_wgs = _samgeo_masks_to_gdf(masks_sorted, img.shape, tile_crs, tile_transform)
        logger.info("  → %d vector features", len(gdf_wgs))

        # Clip to the user's originally drawn area (remove PDOK-padding expansion)
        t2 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
        ox1, oy1 = t2.transform(original_bbox_rd[0], original_bbox_rd[1])
        ox2, oy2 = t2.transform(original_bbox_rd[2], original_bbox_rd[3])
        clip_box = shapely_box(ox1, oy1, ox2, oy2)
        gdf_wgs  = gdf_wgs[gdf_wgs.geometry.intersects(clip_box)].copy()
        gdf_wgs["geometry"] = gdf_wgs.geometry.intersection(clip_box)
        gdf_wgs  = gdf_wgs[~gdf_wgs.geometry.is_empty].reset_index(drop=True)
        logger.info("  → %d features after bbox clip", len(gdf_wgs))

        # CLIP classify all masks (same order as masks_sorted)
        logger.info("  → CLIP classifying %d segments …", len(masks_sorted))
        clip_res = _clip_classify_masks(img, masks_sorted)

        # Build records
        records = []
        for _, row in gdf_wgs.iterrows():
            mask_idx = int(row["mask_id"]) - 1   # 1-indexed → 0-indexed
            if mask_idx < 0 or mask_idx >= len(masks_sorted):
                continue
            mdata = masks_sorted[mask_idx]
            cr    = clip_res[mask_idx] if mask_idx < len(clip_res) else {"top_idx": [0], "top_scores": [0.0]}

            label_idx = cr["top_idx"][0]
            label     = CLIP_LABELS[label_idx]
            score     = cr["top_scores"][0]
            is_constr = label_idx in CONSTRUCTION_IDXS
            color     = LABEL_COLORS_LIST[label_idx] if label_idx < len(LABEL_COLORS_LIST) else "#aaaaaa"
            top3      = [{"label": CLIP_LABELS[i], "score": round(float(s), 3)}
                         for i, s in zip(cr["top_idx"], cr["top_scores"])]

            records.append({
                "geometry":        row.geometry,
                "label":           label,
                "clip_score":      round(float(score), 3),
                "is_construction": is_constr,
                "area_m2":         round(float(mdata["area"]), 1),
                "stability_score": round(float(mdata.get("stability_score", 0)), 3),
                "predicted_iou":   round(float(mdata.get("predicted_iou",   0)), 3),
                "top3_labels":     " | ".join(t["label"] for t in top3),
                "top3_scores":     " | ".join(f"{t['score']:.2f}" for t in top3),
                "_color":          color,
                "_top3":           top3,
            })

        if not records:
            return {"geojson": {"type": "FeatureCollection", "features": []},
                    "bounds":  [list(_to_wgs84_bounds(original_bbox_rd)[1::-1]),
                                list(_to_wgs84_bounds(original_bbox_rd)[3:1:-1])],
                    "stats":   {"total": 0, "construction": 0, "labels": {}}}

        gdf_out   = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        gpkg_out  = ROOT.parent / "outputs" / f"segments_{year}.gpkg"
        gpkg_out.parent.mkdir(exist_ok=True)
        gpkg_cols = [c for c in gdf_out.columns if not c.startswith("_")]
        gdf_out[gpkg_cols].to_file(str(gpkg_out), driver="GPKG", layer="segments")
        LAST_GPKG = gpkg_out
        logger.info("GeoPackage saved → %s", gpkg_out)

        features_json = [{
            "type": "Feature",
            "geometry": rec["geometry"].__geo_interface__,
            "properties": {
                "label":           rec["label"],
                "score":           rec["clip_score"],
                "color":           rec["_color"],
                "is_construction": rec["is_construction"],
                "top3":            rec["_top3"],
                "area_m2":         rec["area_m2"],
                "stability_score": rec["stability_score"],
                "predicted_iou":   rec["predicted_iou"],
            },
        } for rec in records]

        # Construction segments drawn on top
        features_json.sort(key=lambda f: (0 if f["properties"]["is_construction"] else 1,
                                          -f["properties"]["area_m2"]))
        n_constr = sum(1 for f in features_json if f["properties"]["is_construction"])
        label_counts: dict = {}
        for f in features_json:
            lbl = f["properties"]["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        west, south, east, north = _to_wgs84_bounds(original_bbox_rd)
        return {
            "geojson": {"type": "FeatureCollection", "features": features_json},
            "bounds":  [[south, west], [north, east]],
            "stats":   {"total": len(features_json), "construction": n_constr, "labels": label_counts},
        }


# ── Dense detection ────────────────────────────────────────────────────────────
def _run_detection(bbox_rd: tuple, threshold: float, year_a: int, year_b: int) -> dict:
    global LAST_GPKG
    import geopandas as gpd
    from rasterio.features import rasterize as rio_rasterize, shapes as rio_shapes
    from rasterio.transform import from_bounds
    from shapely.geometry import shape

    tile_size, overlap = 1024, 128
    transforms = get_inference_transforms(tile_size)

    with tempfile.TemporaryDirectory() as tmp:
        bbox_rd, path_a, path_b = _download_pair(bbox_rd, year_a, year_b, tmp)

        img_a, src_a = load_geotiff_rgb(path_a)
        img_b, src_b = load_geotiff_rgb(path_b)
        src_a.close(); src_b.close()
        H, W = img_b.shape[:2]
        rd_transform = from_bounds(*bbox_rd, width=W, height=H)

        gdf_m = gpd.read_file(str(ROOT.parent / "data/raw/BT2022.gpkg"), bbox=bbox_rd)
        if len(gdf_m) and str(gdf_m.crs) != "EPSG:28992":
            gdf_m = gdf_m.to_crs("EPSG:28992")
        geoms = [g for g in gdf_m.geometry if g and not g.is_empty] if len(gdf_m) else []
        mask_2022 = (
            rio_rasterize([(g, 1) for g in geoms], out_shape=(H, W),
                          transform=rd_transform, fill=0, dtype="uint8").astype(np.float32)
            if geoms else np.zeros((H, W), dtype=np.float32)
        )

        tiles_b, coords = tile_image_for_inference(img_b, tile_size, overlap)
        tiles_a, _      = tile_image_for_inference(img_a, tile_size, overlap)
        masks_a, _      = tile_image_for_inference(
            (mask_2022 * 255).astype(np.uint8)[:, :, None].repeat(3, 2), tile_size, overlap
        )

        # Optional depth + NDVI signals at full image resolution
        import cv2
        depth_full = _compute_depth_signal(img_b)
        if depth_full is not None:
            logger.info("Depth signal ready.")
        cir_img   = _fetch_cir_tile(bbox_rd, year_b)
        ndvi_full = _compute_ndvi_signal(cir_img, img_b.shape) if cir_img is not None else None
        if ndvi_full is not None:
            logger.info("NDVI signal ready.")

        combined_t, pre_t, new_t = [], [], []
        for (tb, ta, ma), (x0, y0, x1, y1) in zip(zip(tiles_b, tiles_a, masks_a), coords):
            res = transforms(image=tb, image2=ta,
                             mask=np.zeros(tb.shape[:2], dtype=np.uint8),
                             mask2=np.zeros(ta.shape[:2], dtype=np.uint8))
            t_b = res["image"].unsqueeze(0).to(DEVICE)
            t_a = res["image2"].unsqueeze(0).to(DEVICE)
            m_a = torch.from_numpy((ma[:, :, 0] > 127).astype(np.float32)
                                   ).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                ea = F.normalize(MODEL.image_encoder(t_a)["vision_features"], dim=1)
                eb = F.normalize(MODEL.image_encoder(t_b)["vision_features"], dim=1)
                pn = F.normalize(PROTOTYPES, dim=1)
                Hf, Wf = eb.shape[2:]
                b_f = eb.squeeze(0).permute(1, 2, 0).reshape(-1, eb.shape[1])
                a_f = ea.squeeze(0).permute(1, 2, 0).reshape(-1, ea.shape[1])
                up  = dict(size=(tile_size, tile_size), mode="bilinear", align_corners=False)
                # SAM2 prototype appearance score
                sam2_appear = F.interpolate(((b_f @ pn.T).max(1).values
                                        .reshape(1, 1, Hf, Wf).clamp(-1, 1).add(1).div(2)), **up)
                # CLIP spatial score (4×4 grid on original tile_b)
                clip_map = torch.from_numpy(_clip_spatial_score(tb)).unsqueeze(0).unsqueeze(0).to(DEVICE)
                clip_map = F.interpolate(clip_map, size=(tile_size, tile_size),
                                         mode="bilinear", align_corners=False)
                # Optional depth + NDVI tile slices
                extra_maps = []
                if depth_full is not None:
                    d_crop = depth_full[y0:y1, x0:x1]
                    d_t    = torch.from_numpy(
                        cv2.resize(d_crop, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
                    ).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    extra_maps.append(d_t)
                if ndvi_full is not None:
                    n_crop = ndvi_full[y0:y1, x0:x1]
                    n_t    = torch.from_numpy(
                        cv2.resize(n_crop, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
                    ).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    extra_maps.append(n_t)
                # Blend: base 50/50 SAM2+CLIP, each extra signal adds 10% (reducing base proportionally)
                n_extra   = len(extra_maps)
                base_w    = 1.0 - 0.10 * n_extra   # 0.50 or 0.40 depending on signals
                half_base = base_w / 2.0
                appear    = half_base * sam2_appear + half_base * clip_map
                for em in extra_maps:
                    appear = appear + 0.10 * em
                appear = appear.clamp(0, 1)

                change = F.interpolate(((1 - (b_f * a_f).sum(1).clamp(-1, 1))
                                        .reshape(1, 1, Hf, Wf).div(2)), **up)
                pre  = appear * m_a
                new_ = appear * change * (1 - m_a)
                comb = (pre + new_).clamp(0, 1)
            combined_t.append(comb.squeeze().cpu().numpy())
            pre_t.append(pre.squeeze().cpu().numpy())
            new_t.append(new_.squeeze().cpu().numpy())

        hw = (H, W)
        prob_c = stitch_tiles(combined_t, coords, hw)
        prob_p = stitch_tiles(pre_t,      coords, hw)
        prob_n = stitch_tiles(new_t,      coords, hw)

        binary  = (prob_c >= threshold).astype(np.uint8)
        records = []
        for geom_j, val in rio_shapes(binary, mask=binary, transform=rd_transform):
            if not val: continue
            poly = shape(geom_j)
            if poly.is_empty or poly.area < 4.0: continue
            ys, xs = np.where(binary > 0)
            mp = float(prob_p[ys, xs].mean()) if len(ys) else 0.0
            mn = float(prob_n[ys, xs].mean()) if len(ys) else 0.0
            records.append({
                "geometry":         poly,
                "type":             "preexisting" if mp >= mn else "new",
                "prob_combined":    round(float(prob_c[ys, xs].mean()), 4) if len(ys) else 0.0,
                "prob_preexisting": round(mp, 4),
                "prob_new":         round(mn, 4),
                "area_m2":          round(poly.area, 1),
            })

        if records:
            import geopandas as gpd
            gdf_out = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:28992")
            gpkg_path = ROOT.parent / "outputs/dashboard_results.gpkg"
            gpkg_path.parent.mkdir(exist_ok=True)
            gdf_out.to_file(str(gpkg_path), driver="GPKG", layer="construction_sites")
            LAST_GPKG = gpkg_path
            geojson = json.loads(gdf_out.to_crs("EPSG:4326").to_json())
        else:
            geojson = {"type": "FeatureCollection", "features": []}

        west, south, east, north = _to_wgs84_bounds(bbox_rd)
        return {
            "geojson": geojson,
            "raster": {
                "data":   "data:image/png;base64," + base64.b64encode(_prob_to_rgba(prob_c)).decode(),
                "bounds": [[south, west], [north, east]],
            },
            "stats": {
                "n_preexisting": sum(1 for r in records if r["type"] == "preexisting"),
                "n_new":         sum(1 for r in records if r["type"] == "new"),
                "total":         len(records),
                "max_prob":      round(float(prob_c.max()), 3),
            },
        }




# ── One-shot encoding ──────────────────────────────────────────────────────────
def _run_encode_reference(bbox_rd: tuple, year: int) -> dict:
    """Segment-aware one-shot encoding.

    1. SAM2 auto-segment the reference tile.
    2. CLIP classify every segment → keep construction-class segments.
    3. Extract SAM2 features **only from construction pixels** → focused embedding.
    4. Thumbnail shows the reference image with construction segments highlighted.
    """
    global ONE_SHOT_ENCODING, ONE_SHOT_CLIP, ONE_SHOT_PREVIEW

    bbox_rd = _enforce_min_extent(bbox_rd)
    with tempfile.TemporaryDirectory() as tmp:
        dl_8  = PDOKDownloader(resolution="8cm",  crs="EPSG:28992", request_delay=0)
        dl_25 = PDOKDownloader(resolution="25cm", crs="EPSG:28992", request_delay=0)
        tile_path = str(Path(tmp) / f"ref_{year}.tif")
        if _fetch_tile(dl_8, dl_25, bbox_rd, year, tile_path) is None:
            raise HTTPException(404, f"No PDOK coverage for {year}.")

        img, src = load_geotiff_rgb(tile_path)
        src.close()
        H, W = img.shape[:2]

        # 1. Auto-segment (descending sort for full coverage)
        logger.info("One-shot encode: SAM2 auto-segment reference tile …")
        masks_data = SAM2_MASK_GEN.generate(img)
        if not masks_data:
            raise HTTPException(422, "SAM2 found no segments in the reference area.")
        masks_sorted = sorted(masks_data, key=lambda m: m["area"], reverse=True)
        logger.info("  → %d segments", len(masks_sorted))

        # 2. CLIP classify — find construction segments
        clip_res = _clip_classify_masks(img, masks_sorted)
        constr_masks = [m for m, cr in zip(masks_sorted, clip_res)
                        if cr["top_idx"][0] in CONSTRUCTION_IDXS]

        # Fallback: take top-5 by best construction label score if none found
        if not constr_masks:
            def _best_constr(cr):
                return max((s for i, s in zip(cr["top_idx"], cr["top_scores"])
                            if i in CONSTRUCTION_IDXS), default=0.0)
            pairs = sorted(zip(masks_sorted, clip_res), key=lambda p: _best_constr(p[1]), reverse=True)
            constr_masks = [m for m, _ in pairs[:5]]

        logger.info("  → %d construction segment(s) identified for encoding", len(constr_masks))

        # 3. Build combined construction mask
        fg_mask = np.zeros((H, W), dtype=bool)
        for m in constr_masks:
            fg_mask |= m["segmentation"]
        if fg_mask.sum() < 64:
            fg_mask = np.ones((H, W), dtype=bool)  # full fallback

        # 4. SAM2 encode → extract features from construction pixels only
        tfm = get_inference_transforms(1024)
        res = tfm(image=img, mask=np.zeros((H, W), dtype=np.uint8),
                  image2=img, mask2=np.zeros((H, W), dtype=np.uint8))
        t = res["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feats_n = F.normalize(MODEL.image_encoder(t)["vision_features"], dim=1)
            ONE_SHOT_ENCODING = _segment_mean_features(feats_n, fg_mask)

        # 5. CLIP encode construction crops
        constr_crops = []
        for m in constr_masks[:8]:
            x, y, bw, bh = [int(v) for v in m["bbox"]]
            crop = img[y: y + bh, x: x + bw]
            if crop.size > 0:
                constr_crops.append(Image.fromarray(crop))
        if constr_crops:
            with torch.no_grad():
                inp    = CLIP_PROCESSOR(images=constr_crops, return_tensors="pt", padding=True).to(DEVICE)
                v_out  = CLIP_MODEL.vision_model(pixel_values=inp["pixel_values"])
                feats  = F.normalize(CLIP_MODEL.visual_projection(v_out.pooler_output), dim=-1)
                ONE_SHOT_CLIP = F.normalize(feats.mean(dim=0), dim=0)

        # 6. Thumbnail: original image + construction segments highlighted in orange
        thumb_arr = img.copy()
        overlay   = thumb_arr.copy()
        for m in constr_masks:
            overlay[m["segmentation"]] = [255, 140, 0]
        thumb_arr = (0.55 * thumb_arr + 0.45 * overlay).astype(np.uint8)
        thumb = Image.fromarray(thumb_arr).resize((240, 240), Image.LANCZOS)
        buf   = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=82)
        ONE_SHOT_PREVIEW = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        logger.info("Reference encoded → SAM2 [%d-dim], %d constr segments",
                    ONE_SHOT_ENCODING.shape[0], len(constr_masks))
        west, south, east, north = _to_wgs84_bounds(bbox_rd)
        return {
            "preview":                ONE_SHOT_PREVIEW,
            "bounds":                 [[south, west], [north, east]],
            "n_construction_segments": len(constr_masks),
            "n_total_segments":       len(masks_sorted),
        }


def _run_one_shot_search(bbox_rd: tuple, year: int, threshold: float) -> dict:
    """Segment-level one-shot similarity search.

    For each segment in the search tile:
      - Extract mean SAM2 features from the segment's pixels
      - Cosine similarity to ONE_SHOT_ENCODING  (focused construction embedding)
      - CLIP construction score                  (semantic check)
      - final_score = 0.55 * sam2_sim + 0.45 * clip_norm
    Returns coloured GeoJSON segments (no heatmap needed).
    """
    import geopandas as gpd

    if ONE_SHOT_ENCODING is None:
        raise HTTPException(400, "No reference encoding — call /api/encode-reference first.")

    original_bbox_rd = bbox_rd
    bbox_rd = _enforce_min_extent(bbox_rd)

    with tempfile.TemporaryDirectory() as tmp:
        dl_8  = PDOKDownloader(resolution="8cm",  crs="EPSG:28992", request_delay=0)
        dl_25 = PDOKDownloader(resolution="25cm", crs="EPSG:28992", request_delay=0)
        tile_path = str(Path(tmp) / f"search_{year}.tif")
        if _fetch_tile(dl_8, dl_25, bbox_rd, year, tile_path) is None:
            raise HTTPException(404, f"No PDOK coverage for {year}.")

        img, src = load_geotiff_rgb(tile_path)
        tile_crs       = src.crs
        tile_transform = src.transform
        src.close()
        H, W = img.shape[:2]

        # 1. SAM2 auto-segment search tile (descending sort → full coverage)
        logger.info("One-shot search: SAM2 auto-segment …")
        masks_data   = SAM2_MASK_GEN.generate(img)
        masks_sorted = sorted(masks_data, key=lambda m: m["area"], reverse=True)
        logger.info("  → %d segments", len(masks_sorted))

        # 2. CLIP classify all segments
        clip_res = _clip_classify_masks(img, masks_sorted)

        # 3. SAM2 encode full tile → feature map
        tfm = get_inference_transforms(1024)
        res = tfm(image=img, mask=np.zeros((H, W), dtype=np.uint8),
                  image2=img, mask2=np.zeros((H, W), dtype=np.uint8))
        t = res["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feats_n = F.normalize(MODEL.image_encoder(t)["vision_features"], dim=1)
        ref = ONE_SHOT_ENCODING.to(DEVICE)  # [C]

        # 4. Score each segment
        scored = []
        for m, cr in zip(masks_sorted, clip_res):
            seg_vec  = _segment_mean_features(feats_n, m["segmentation"])
            sam2_sim = float((seg_vec * ref).sum().clamp(0, 1).item())

            # CLIP construction score, normalized to [0,1] (raw scores ~0.1-0.35)
            raw_constr = max(
                (s for i, s in zip(cr["top_idx"], cr["top_scores"]) if i in CONSTRUCTION_IDXS),
                default=0.0,
            )
            clip_norm  = float(min(max(raw_constr - 0.08, 0.0) / 0.22, 1.0))
            final      = 0.55 * sam2_sim + 0.45 * clip_norm

            scored.append({
                "mask":        m,
                "cr":          cr,
                "sam2_sim":    round(sam2_sim,  3),
                "clip_norm":   round(clip_norm, 3),
                "final_score": round(final,     3),
            })

        # 5. Vectorize with same descending-area order
        gdf_wgs = _samgeo_masks_to_gdf(masks_sorted, img.shape, tile_crs, tile_transform)

        # Clip to original drawn bbox
        from shapely.geometry import box as shapely_box
        t2 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
        ox1, oy1 = t2.transform(original_bbox_rd[0], original_bbox_rd[1])
        ox2, oy2 = t2.transform(original_bbox_rd[2], original_bbox_rd[3])
        clip_box = shapely_box(ox1, oy1, ox2, oy2)
        gdf_wgs  = gdf_wgs[gdf_wgs.geometry.intersects(clip_box)].copy()
        gdf_wgs["geometry"] = gdf_wgs.geometry.intersection(clip_box)
        gdf_wgs  = gdf_wgs[~gdf_wgs.geometry.is_empty].reset_index(drop=True)

        # 6. Build GeoJSON — colour by score (red = high similarity)
        features = []
        n_matches = 0
        for _, row in gdf_wgs.iterrows():
            mask_idx = int(row["mask_id"]) - 1
            if mask_idx < 0 or mask_idx >= len(scored):
                continue
            sc = scored[mask_idx]
            cr = sc["cr"]
            fs = sc["final_score"]
            is_match = fs >= threshold

            # Colour: matches in red-orange gradient, non-matches in gray
            if is_match:
                n_matches += 1
                # gradient: threshold=gray → 1.0=red
                t_  = (fs - threshold) / max(1.0 - threshold, 0.01)
                r   = 255
                g   = max(0, int(140 * (1 - t_)))
                b   = 0
                color = f"#{r:02x}{g:02x}{b:02x}"
            else:
                color = "#606060"

            label_idx = cr["top_idx"][0]
            features.append({
                "type": "Feature",
                "geometry": row.geometry.__geo_interface__,
                "properties": {
                    "final_score": fs,
                    "sam2_sim":    sc["sam2_sim"],
                    "clip_norm":   sc["clip_norm"],
                    "label":       CLIP_LABELS[label_idx],
                    "color":       color,
                    "is_match":    is_match,
                    "area_m2":     round(float(sc["mask"]["area"]), 1),
                },
            })

        # Matches on top
        features.sort(key=lambda f: (0 if f["properties"]["is_match"] else 1,
                                     -f["properties"]["final_score"]))

        west, south, east, north = _to_wgs84_bounds(original_bbox_rd)
        return {
            "geojson": {"type": "FeatureCollection", "features": features},
            "bounds":  [[south, west], [north, east]],
            "stats": {
                "n_matches": n_matches,
                "n_total":   len(features),
                "max_score": round(max((f["properties"]["final_score"] for f in features), default=0.0), 3),
            },
            "reference_preview": ONE_SHOT_PREVIEW,
        }


# ── API endpoints ──────────────────────────────────────────────────────────────
@app.post("/api/detect")
async def detect(req: DetectRequest) -> dict:
    if not MODEL_READY:
        raise HTTPException(503, "Model loading — check /api/health and retry.")
    bbox_rd = _bbox_rd_from_wgs84(req.bbox_wgs84)
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=1),
            _run_detection, bbox_rd, req.threshold, req.year_a, req.year_b,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Detection error")
        raise HTTPException(500, str(exc))



@app.post("/api/auto-segment")
async def auto_segment(req: AutoSegRequest) -> dict:
    if not MODEL_READY:
        raise HTTPException(503, "Model loading — check /api/health.")
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


@app.post("/api/encode-reference")
async def encode_reference(req: EncodeReferenceRequest) -> dict:
    if not MODEL_READY:
        raise HTTPException(503, "Model loading — check /api/health.")
    bbox_rd = _bbox_rd_from_wgs84(req.bbox_wgs84)
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=1),
            _run_encode_reference, bbox_rd, req.year,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Encode reference error")
        raise HTTPException(500, str(exc))


@app.post("/api/one-shot-search")
async def one_shot_search(req: OneShotSearchRequest) -> dict:
    if not MODEL_READY:
        raise HTTPException(503, "Model loading — check /api/health.")
    bbox_rd = _bbox_rd_from_wgs84(req.bbox_wgs84)
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=1),
            _run_one_shot_search, bbox_rd, req.year, req.threshold,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("One-shot search error")
        raise HTTPException(500, str(exc))


@app.get("/api/health")
async def health() -> dict:
    return {"ready": MODEL_READY, "device": str(DEVICE) if DEVICE else "loading"}


@app.post("/api/test-bbox")
async def test_bbox(req: DetectRequest) -> dict:
    """Debug: returns the RD bbox + a quick PDOK availability check (no model needed)."""
    import io as _io, requests as _req
    from PIL import Image as _Img
    bbox_rd = _bbox_rd_from_wgs84(req.bbox_wgs84)
    expanded = _enforce_min_extent(bbox_rd)
    results = {}
    for year in (req.year_a, req.year_b):
        for suffix, res_label in (("orthoHR", "8cm"), ("ortho25", "25cm")):
            layer = f"{year}_{suffix}"
            p = {"SERVICE":"WMS","VERSION":"1.3.0","REQUEST":"GetMap",
                 "LAYERS":layer,"STYLES":"","CRS":"EPSG:28992",
                 "BBOX":f"{expanded[0]},{expanded[1]},{expanded[2]},{expanded[3]}",
                 "WIDTH":"256","HEIGHT":"256","FORMAT":"image/png"}
            r = _req.get("https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0", params=p, timeout=15)
            ct = r.headers.get("Content-Type","")
            if "xml" in ct:
                results[f"{year}_{res_label}"] = "XML_ERROR"
            else:
                img = np.array(_Img.open(_io.BytesIO(r.content)).convert("RGB"))
                wf = float((img==255).all(axis=-1).mean())
                results[f"{year}_{res_label}"] = f"white={wf:.3f} {'BLANK' if wf>0.98 else 'OK'}"
    return {"bbox_wgs84": req.bbox_wgs84, "bbox_rd_original": bbox_rd,
            "bbox_rd_expanded": expanded, "pdok": results}


@app.get("/api/download/results.gpkg")
async def download_gpkg() -> FileResponse:
    if LAST_GPKG is None or not LAST_GPKG.exists():
        raise HTTPException(404, "No results yet.")
    return FileResponse(str(LAST_GPKG), media_type="application/geopackage+sqlite3",
                        filename="construction_sites.gpkg")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")
