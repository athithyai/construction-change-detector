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
SAM2_PREDICTOR = None
GDINO_PROCESSOR = None
GDINO_MODEL = None

# Construction-related text prompts for Grounding DINO
CONSTRUCTION_PROMPTS = (
    "crane . tower crane . scaffolding . excavator . bulldozer . dump truck . "
    "concrete mixer . pile driver . construction site . building under construction . "
    "construction machinery . foundation . building frame . steel structure"
)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SamSpade")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC = ROOT / "static"
STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# ── Model loading ──────────────────────────────────────────────────────────────
def _load_model() -> None:
    global MODEL, PROTOTYPES, DEVICE, CFG, MODEL_READY, SAM2_PREDICTOR
    global GDINO_PROCESSOR, GDINO_MODEL
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

        # SAM2ImagePredictor — shares weights with MODEL (no extra VRAM)
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        SAM2_PREDICTOR = SAM2ImagePredictor(model.sam2_model)
        logger.info("SAM2ImagePredictor ready.")

        # Grounding DINO for text → bounding boxes
        logger.info("Loading Grounding DINO …")
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        gdino_id = "IDEA-Research/grounding-dino-base"
        GDINO_PROCESSOR = AutoProcessor.from_pretrained(gdino_id)
        GDINO_MODEL     = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_id).to(device)
        GDINO_MODEL.eval()
        logger.info("Grounding DINO ready.")

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


class SegmentRequest(BaseModel):
    polygon_wgs84: dict        # GeoJSON Polygon geometry
    text_prompts: str = CONSTRUCTION_PROMPTS
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    year_b: int = 2024


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

        combined_t, pre_t, new_t = [], [], []
        for tb, ta, ma in zip(tiles_b, tiles_a, masks_a):
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
                appear = F.interpolate(((b_f @ pn.T).max(1).values
                                        .reshape(1, 1, Hf, Wf).clamp(-1, 1).add(1).div(2)), **up)
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


# ── SAM + text segmentation ────────────────────────────────────────────────────
def _run_segmentation(polygon_geojson: dict, text_prompts: str,
                      box_threshold: float, text_threshold: float, year_b: int) -> dict:
    """Grounding DINO → boxes → SAM2 → masks → score vs prototypes."""
    global LAST_GPKG
    import geopandas as gpd
    from rasterio.transform import from_bounds
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape as shapely_shape, mapping
    from shapely.ops import unary_union

    # ── 1. Get bbox from polygon ──────────────────────────────────────────────
    t_fwd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    coords_wgs = polygon_geojson["coordinates"][0]
    xs_rd = [t_fwd.transform(c[0], c[1])[0] for c in coords_wgs]
    ys_rd = [t_fwd.transform(c[0], c[1])[1] for c in coords_wgs]
    bbox_rd = (min(xs_rd), min(ys_rd), max(xs_rd), max(ys_rd))

    bbox_rd = _enforce_min_extent(bbox_rd)

    with tempfile.TemporaryDirectory() as tmp:
        # ── 2. Download tile ──────────────────────────────────────────────────
        dl_8  = PDOKDownloader(resolution="8cm",  crs="EPSG:28992", request_delay=0)
        dl_25 = PDOKDownloader(resolution="25cm", crs="EPSG:28992", request_delay=0)
        path_b = str(Path(tmp) / f"tile_{year_b}.tif")
        logger.info("Fetching %d tile for segmentation  bbox_rd=%s …", year_b, bbox_rd)
        r = _fetch_tile(dl_8, dl_25, bbox_rd, year_b, path_b)
        if r is None or not Path(path_b).exists():
            raise HTTPException(404, f"No PDOK coverage for year {year_b} at this location. bbox_rd={bbox_rd}")

        img_b, src_b = load_geotiff_rgb(path_b)   # HxWx3 uint8
        src_b.close()
        H, W = img_b.shape[:2]
        rd_transform = from_bounds(*bbox_rd, width=W, height=H)
        pil_img = Image.fromarray(img_b)

        # ── 3. Grounding DINO: text → boxes ──────────────────────────────────
        logger.info("Running Grounding DINO: '%s'", text_prompts[:60])
        inputs = GDINO_PROCESSOR(
            images=pil_img,
            text=text_prompts,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = GDINO_MODEL(**inputs)

        results = GDINO_PROCESSOR.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(H, W)],
        )[0]

        boxes  = results["boxes"].cpu().numpy()   # [N, 4] in xyxy pixel coords
        labels = results["labels"]
        scores = results["scores"].cpu().numpy()

        logger.info("Grounding DINO found %d objects.", len(boxes))

        if len(boxes) == 0:
            return {
                "geojson": {"type": "FeatureCollection", "features": []},
                "stats": {"total": 0, "objects": []},
                "message": "No construction objects detected. Try lowering the threshold or zoom in more.",
            }

        # ── 4. SAM2: boxes → precise masks ───────────────────────────────────
        logger.info("Running SAM2 predictor on %d boxes…", len(boxes))
        SAM2_PREDICTOR.set_image(img_b)

        all_masks, all_labels, all_scores = [], [], []
        for box, label, score in zip(boxes, labels, scores):
            masks, mask_scores, _ = SAM2_PREDICTOR.predict(
                box=box,
                multimask_output=False,
            )
            best = int(np.argmax(mask_scores))
            all_masks.append(masks[best].astype(bool))   # HxW bool
            all_labels.append(label)
            all_scores.append(float(score))

        # ── 5. Score each mask vs construction prototypes ─────────────────────
        transforms = get_inference_transforms(1024)
        res = transforms(image=img_b,
                         mask=np.zeros(img_b.shape[:2], dtype=np.uint8),
                         image2=img_b,
                         mask2=np.zeros(img_b.shape[:2], dtype=np.uint8))
        t_b = res["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feats = MODEL.image_encoder(t_b)["vision_features"]   # [1,C,Hf,Wf]
            feats = F.normalize(feats, dim=1)
            pn    = F.normalize(PROTOTYPES, dim=1)                 # [K,C]
            Hf, Wf = feats.shape[2:]
            # Per-pixel max cosine sim → appearance score
            f_flat  = feats.squeeze(0).permute(1, 2, 0).reshape(-1, feats.shape[1])
            sim_map = (f_flat @ pn.T).max(1).values.reshape(Hf, Wf)
            sim_map = ((sim_map.clamp(-1, 1) + 1) / 2).cpu().numpy()  # [Hf, Wf] in [0,1]

        # Upsample sim_map to full tile resolution
        import cv2
        sim_full = cv2.resize(sim_map, (W, H), interpolation=cv2.INTER_LINEAR)

        # ── 6. Build GeoJSON features ─────────────────────────────────────────
        records = []
        t2 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

        for mask, label, det_score in zip(all_masks, all_labels, all_scores):
            # Prototype score for this mask
            mask_u8 = mask.astype(np.uint8)
            proto_score = float(sim_full[mask].mean()) if mask.any() else 0.0

            # Vectorise the mask
            polys = []
            for geom_j, val in rio_shapes(mask_u8, mask=mask_u8, transform=rd_transform):
                if val:
                    p = shapely_shape(geom_j)
                    if not p.is_empty and p.area > 1.0:
                        polys.append(p)
            if not polys:
                continue
            poly_rd = unary_union(polys)

            # Convert to WGS84
            from shapely.ops import transform as shp_transform
            poly_wgs = shp_transform(
                lambda x, y, z=None: t2.transform(x, y),
                poly_rd,
            )

            records.append({
                "type": "Feature",
                "geometry": mapping(poly_wgs),
                "properties": {
                    "label":       label,
                    "det_score":   round(det_score, 3),
                    "proto_score": round(proto_score, 3),
                    "combined":    round((det_score + proto_score) / 2, 3),
                    "area_m2":     round(poly_rd.area, 1),
                }
            })

        records.sort(key=lambda r: r["properties"]["combined"], reverse=True)

        # Save GPKG
        if records:
            import geopandas as gpd
            gdf_out = gpd.GeoDataFrame.from_features(records, crs="EPSG:4326")
            gpkg_path = ROOT.parent / "outputs/dashboard_segments.gpkg"
            gpkg_path.parent.mkdir(exist_ok=True)
            gdf_out.to_file(str(gpkg_path), driver="GPKG", layer="construction_objects")
            LAST_GPKG = gpkg_path

        label_counts = {}
        for r in records:
            l = r["properties"]["label"]
            label_counts[l] = label_counts.get(l, 0) + 1

        return {
            "geojson": {"type": "FeatureCollection", "features": records},
            "stats": {
                "total":    len(records),
                "objects":  label_counts,
                "max_combined": round(max((r["properties"]["combined"] for r in records), default=0), 3),
            },
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


@app.post("/api/segment")
async def segment(req: SegmentRequest) -> dict:
    if not MODEL_READY:
        raise HTTPException(503, "Model loading — check /api/health and retry.")
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=1),
            _run_segmentation,
            req.polygon_wgs84, req.text_prompts,
            req.box_threshold, req.text_threshold, req.year_b,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Segmentation error")
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
