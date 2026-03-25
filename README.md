<div align="center">

# 🕵️ SamSpade

### *SAM2 meets a shovel. Barren land doesn't stand a chance.*

**Dense SAM2 segmentation · NDVI · Depth · CLIP — construction-terrain analysis over the Netherlands**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![SAM2](https://img.shields.io/badge/SAM2-Hiera--L-0064FF?style=flat-square)](https://github.com/facebookresearch/sam2)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--L/14-412991?style=flat-square)](https://huggingface.co/openai/clip-vit-large-patch14)
[![PDOK](https://img.shields.io/badge/Data-PDOK%20orthoHR-00A550?style=flat-square)](https://service.pdok.nl)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## What is this?

SamSpade identifies **barren land, exposed soil, and disturbed terrain** that looks like
construction — purely from aerial imagery, with zero labels required.

Draw a bounding box on the map, and SamSpade:

1. Fetches **2022 and 2024 RGB + CIR** tiles from [PDOK](https://service.pdok.nl)
2. Computes **NDVI** for both years (bare soil = low NDVI)
3. Computes **depth maps** using Depth Anything V2 (rough terrain = high variance)
4. Runs **dense SAM2 auto-segmentation** — every pixel gets a segment
5. Classifies every segment using **CLIP + NDVI + depth roughness**
6. Returns a fully interactive map with colour-coded terrain polygons

---

## Interactive Dashboard

### Run it

```bash
# 1. Activate the environment
conda activate construction

# 2. Start the server
python -m uvicorn dashboard.app:app --host 0.0.0.0 --port 8000

# 3. Open in browser
# http://localhost:8000
```

> **First load:** SAM2 Hiera-L + CLIP ViT-L/14 + Depth Anything V2 take ~60–90 s to load.
> The status bar shows `Models ready · cuda` when done.

### Workflow

```
1.  Draw a bbox on the map
2.  Press ▶ Analyze Area
3.  Results appear in ~30–120 s depending on tile size
```

The analysis always covers **both 2022 and 2024** for direct comparison.

### What you see

```
┌──────────────────────────────────────────────────────────────────┐
│  MAP (Leaflet)                           │  SIDEBAR              │
│                                          │                       │
│  Base imagery (radio):                   │  Base Imagery:        │
│  • RGB 2022  • CIR 2022                 │  RGB 2022 | CIR 2022  │
│  • RGB 2024  • CIR 2024                 │  RGB 2024 | CIR 2024  │
│  • Live PDOK tile (default)              │  Live PDOK            │
│                                          │                       │
│  Overlay toggles:                        │  Derived Overlays:    │
│  • NDVI 2022 / 2024 (RdYlGn)           │  NDVI 2022 | 2024     │
│  • Depth 2022 / 2024 (inferno)          │  Depth 2022 | 2024    │
│                                          │                       │
│  Analysis layers:                        │  Analysis Layers:     │
│  • All SAM segments (terrain-coloured)   │  ▪ SAM segments       │
│  • Construction terrain only (bold)      │  ▪ Construction only  │
│  • 2022 reference sites (orange dashed) │  ▪ 2022 reference     │
│                                          │                       │
│  Legend (bottom-right): terrain classes  │  Draw Area            │
│  Gradient scale (bottom-left): NDVI/Depth│  ▶ Analyze            │
└──────────────────────────────────────────────────────────────────┘
```

### Terrain classes

| Colour | Class |
|---|---|
| 🔴 `#FF4500` | **likely construction terrain** |
| 🟫 `#CD853F` | exposed soil / bare ground |
| 🟢 `#22BB44` | vegetation |
| 🔵 `#1E90FF` | water |
| 🟣 `#9370DB` | roof / building |
| ⬜ `#778899` | paved surface |
| ⬛ `#505060` | shadow / unknown |

### Segment popup (click any polygon)

Each segment shows:
- Terrain label + colour
- Construction score (0–100 %) with visual bar
- NDVI 2022 and NDVI 2024 mean (with vegetation/bare tag)
- Depth mean and roughness
- Top-3 CLIP semantic matches with scores
- Area (px) and SAM IoU

### Download results

Click **⬇ Download GeoPackage** in the sidebar, or:

```bash
curl http://localhost:8000/api/download/results.gpkg -o segments_analysis.gpkg
```

Load in QGIS: `Layer → Add Vector Layer → segments_analysis.gpkg`

---

## How the Analysis Works

### 1 — Imagery acquisition

PDOK serves 8 cm orthoHR and Color-Infrared imagery.
Both 2022 and 2024 are fetched automatically for every bbox.

```
PDOK RGB WMS  →  rgb_2022.tif,  rgb_2024.tif
PDOK CIR WMS  →  cir_2022 arr, cir_2024 arr  (R=NIR, G=Red, B=Green)
```

> **PDOK minimum bbox:** 600 × 600 m to avoid blank tiles.
> The server expands internally; results are clipped back to the exact drawn bbox.

### 2 — NDVI

```
NDVI = (NIR − Red) / (NIR + Red)
```

| NDVI range | Terrain |
|---|---|
| < 0.05 | bare soil / construction |
| 0.05–0.30 | mixed / sparse vegetation |
| > 0.30 | healthy vegetation |

Coloured with **RdYlGn** matplotlib colormap (red=bare, green=vegetation).

### 3 — Depth

Depth Anything V2 Small runs on each year's RGB tile → normalised `[0, 1]` depth map.

Coloured with **inferno** matplotlib colormap.

Per-segment **depth variance** = terrain roughness signal.
Construction sites → disturbed ground → high variance.

### 4 — Dense SAM2 segmentation

`SAM2AutomaticMaskGenerator` with `points_per_side=32` runs on the **2024 RGB** tile.

Key: masks are sorted **descending by area** before rasterising into a label map.
Large masks fill first; small masks overwrite them → every pixel gets its
**smallest enclosing segment**. Full bbox coverage guaranteed.

### 5 — Per-segment classification

For each SAM segment:

```python
# CLIP classify the cropped segment
clip_sims = CLIP(segment_crop, 14_aerial_labels)

# Derived signals
ndvi_score    = clip((0.2 − mean_ndvi) / 0.5, 0, 1)   # low NDVI → high
roughness     = clip(depth_variance / 0.025,  0, 1)   # high var → high
clip_constr   = clip((max_constr_sim − 0.08) / 0.22, 0, 1)

# Fused construction score
construction_score = 0.40 × ndvi_score
                   + 0.35 × roughness
                   + 0.25 × clip_constr
```

Label assignment (priority order):
1. `construction_score ≥ 0.60` → **likely construction terrain**
2. CLIP water similarity > 0.22 → **water**
3. CLIP building similarity > 0.22 → **roof / building**
4. CLIP paved similarity > 0.22 → **paved surface**
5. CLIP vegetation > 0.22 or NDVI > 0.35 → **vegetation**
6. `construction_score ≥ 0.35` → **exposed soil / bare ground**
7. Everything else → **shadow / unknown**

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/analyze-bbox` | `POST` | Main analysis — see schema below |
| `/api/bt2022` | `GET` | BT2022 construction polygons (GeoJSON) |
| `/api/health` | `GET` | `{"ready": bool, "device": str}` |
| `/api/download/results.gpkg` | `GET` | Download last GeoPackage |

### `POST /api/analyze-bbox`

Request:
```json
{ "bbox_wgs84": [west, south, east, north] }
```

Response:
```json
{
  "bounds":   [[south, west], [north, east]],
  "overlays": {
    "rgb_2022":   "data:image/png;base64,…",
    "rgb_2024":   "data:image/png;base64,…",
    "cir_2022":   "data:image/png;base64,…",
    "cir_2024":   "data:image/png;base64,…",
    "ndvi_2022":  "data:image/png;base64,…",
    "ndvi_2024":  "data:image/png;base64,…",
    "depth_2022": "data:image/png;base64,…",
    "depth_2024": "data:image/png;base64,…"
  },
  "segments": { "type": "FeatureCollection", "features": [...] },
  "stats": {
    "total": 142,
    "construction": 23,
    "labels": { "likely construction terrain": 23, "vegetation": 61, ... }
  }
}
```

Each GeoJSON feature has these properties:

| Property | Type | Description |
|---|---|---|
| `terrain_label` | string | One of 7 terrain classes |
| `construction_score` | float | 0–1 fused construction probability |
| `color` | hex string | Display colour |
| `mean_ndvi_2022` | float \| null | Mean NDVI over segment, 2022 |
| `mean_ndvi_2024` | float \| null | Mean NDVI over segment, 2024 |
| `depth_mean` | float \| null | Mean depth (2024) |
| `depth_roughness` | float \| null | Depth variance × 1000 |
| `area_px` | int | Segment area in pixels |
| `top3_clip` | list | Top-3 CLIP label + score |
| `predicted_iou` | float | SAM2 predicted IoU |
| `stability_score` | float | SAM2 stability score |

---

## Models

| Model | Role | VRAM |
|---|---|---|
| SAM2 Hiera-L | Dense auto-segmentation | ~6 GB |
| CLIP ViT-L/14 | Per-segment semantic classification | ~1.5 GB |
| Depth Anything V2 Small | Terrain depth + roughness | ~0.3 GB |

Total: ~8 GB VRAM. Runs on a single GPU.
CPU fallback available (much slower, ~5–10 min per bbox).

---

## Installation

```bash
# 1. Clone
git clone https://github.com/athithyai/construction-change-detector.git
cd construction-change-detector

# 2. Create environment
conda create -n construction python=3.11
conda activate construction

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) place BT2022.gpkg reference layer
# Put data/raw/BT2022.gpkg to see 2022 construction site overlay on the map

# 5. Start
python -m uvicorn dashboard.app:app --host 0.0.0.0 --port 8000
```

---

## Data Sources

| | Source | Detail |
|---|---|---|
| 🛰️ **RGB Imagery** | [PDOK Luchtfoto WMS](https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0) | 8 cm orthoHR, layers `2022_ortho25` / `2024_ortho25` |
| 🌿 **CIR Imagery** | [PDOK CIR WMS](https://service.pdok.nl/hwh/luchtfotocir/wms/v1_0) | Color-infrared for NDVI, `{year}_ortho25` |
| 🗺️ **Reference polygons** | BT2022.gpkg | 3 664 construction sites from Dutch land-use survey, EPSG:28992 |

<details>
<summary>🔧 PDOK WMS gotchas</summary>

| Gotcha | What happens | Fix |
|---|---|---|
| Small bbox | Returns pure-white blank tile | Enforce 600 m minimum extent centred on drawn area |
| `TIME` parameter | Silently ignored | Use year-specific layer names: `2022_ortho25` |
| `owslib.getmap` | Inconsistent | Use `requests.Session` directly |
| 8 cm blank tile | Some areas only have 25 cm | Auto-fallback: try 8 cm → fall back to 25 cm |

</details>

---

## Project Structure

```
SamSpade/
├── 📁 dashboard/
│   ├── app.py                 # FastAPI backend — all models + /api/analyze-bbox
│   └── static/index.html      # Leaflet.js frontend — single draw-and-run workflow
├── 📁 data/
│   ├── pdok_downloader.py     # PDOK WMS client (RGB + CIR)
│   ├── dataset.py             # Dataset utilities
│   └── transforms.py          # Augmentation pipeline
├── 📁 models/
│   └── feature_utils.py       # Tiling, stitching, GeoTIFF I/O
├── 📁 scripts/
│   ├── download_pdok.py        # Bulk tile downloader
│   └── score_area.py           # CLI: score a bbox → GeoTIFF output
└── 📁 outputs/                 # segments_analysis.gpkg saved here
```

---

<div align="center">

*Named after Sam Spade — the detective who always finds what's hidden.*
*Powered by SAM2 — the model that segments anything.*
*Built for the Netherlands — where everything is always under construction.*

</div>
