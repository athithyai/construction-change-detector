<div align="center">

# 🕵️ SamSpade

**Construction-terrain analysis from aerial imagery — Netherlands**

Draw a bounding box. Get dense SAM2 segments with NDVI, depth, and CLIP classification for 2022 and 2024.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![SAM2](https://img.shields.io/badge/SAM2-Hiera--L-0064FF?style=flat-square)](https://github.com/facebookresearch/sam2)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--L/14-412991?style=flat-square)](https://huggingface.co/openai/clip-vit-large-patch14)
[![PDOK](https://img.shields.io/badge/Imagery-PDOK%208cm-00A550?style=flat-square)](https://service.pdok.nl)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## What it does

Draw a bounding box anywhere in the Netherlands. SamSpade:

1. Fetches **2022 and 2024 RGB + CIR** orthophotos from PDOK (free, 8 cm)
2. Computes **NDVI** for both years — low NDVI reveals bare soil and construction
3. Runs **Depth Anything V2** on both years — high variance reveals disturbed terrain
4. Runs **dense SAM2 auto-segmentation** on the 2024 tile — every pixel gets a segment
5. Classifies each segment using **CLIP + NDVI + depth roughness** into 7 terrain classes
6. Returns an interactive map with all layers switchable, and per-segment metrics in a popup

The goal is to find **barren, exposed-soil, disturbed terrain** that looks like active construction.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/athithyai/construction-change-detector.git
cd construction-change-detector

# 2. Create environment
conda create -n construction python=3.11
conda activate construction

# 3. Install
pip install -r requirements.txt
pip install fastapi uvicorn

# 4. Start the dashboard
python -m uvicorn dashboard.app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**

> Models load on first startup: SAM2 Hiera-L + CLIP ViT-L/14 + Depth Anything V2 Small.
> Takes **60–90 seconds**. The status bar at the bottom of the sidebar shows
> `Models ready · cuda` when done.

---

## Dashboard

### Workflow

```
1.  Press  ✏ Draw bbox on map
2.  Drag a rectangle over any area in the Netherlands
3.  Press  ▶ Analyze Area
4.  Wait ~30–120 s (tile download + SAM2 + CLIP)
5.  All layers appear automatically
```

### Layout

```
┌──────────────────────────────────────────────────────┬──────────────────────┐
│                                                      │  ⬡ SamSpade          │
│                  MAP (Leaflet)                       │                      │
│                                                      │  Base Imagery        │
│  Switchable base imagery:                            │  RGB 2022 │ CIR 2022 │
│  RGB 2022 / CIR 2022 / RGB 2024 / CIR 2024 / PDOK   │  RGB 2024 │ CIR 2024 │
│                                                      │  Live PDOK           │
│  Toggleable overlays:                                │                      │
│  NDVI 2022/2024 (RdYlGn)                            │  Derived Overlays    │
│  Depth 2022/2024 (inferno)                          │  NDVI 2022 │ 2024    │
│                                                      │  Depth 2022 │ 2024   │
│  Analysis layers:                                    │                      │
│  • All SAM segments (terrain-coloured)               │  Analysis Layers     │
│  • Construction terrain only (bold highlight)        │  ▪ SAM segments      │
│  • 2022 reference sites (orange dashed)             │  ▪ Construction only  │
│                                                      │  ▪ 2022 ref sites    │
│  Legend: terrain classes (bottom-right)              │                      │
│  Gradient scale: NDVI / Depth (bottom-left)          │  ✏ Draw bbox         │
│                                                      │  ▶ Analyze           │
│  📡 2024 RGB    🟠 2022 construction sites           │                      │
│                                                      │  Results + stats     │
│                                                      │  ⬇ Download GPKG     │
└──────────────────────────────────────────────────────┴──────────────────────┘
```

### Terrain classes

| Colour | Class | When assigned |
|---|---|---|
| 🔴 `#FF4500` | **likely construction terrain** | construction score ≥ 0.60 |
| 🟫 `#CD853F` | exposed soil / bare ground | score 0.35–0.59, or CLIP bare-soil > 0.20 |
| 🟢 `#22BB44` | vegetation | CLIP veg > 0.22 or NDVI > 0.35 |
| 🔵 `#1E90FF` | water | CLIP water > 0.22 |
| 🟣 `#9370DB` | roof / building | CLIP building > 0.22 |
| ⬜ `#778899` | paved surface | CLIP road/parking > 0.22 |
| ⬛ `#505060` | shadow / unknown | none of the above |

### Segment popup

Click any polygon to see:

```
┌─────────────────────────────────────────────┐
│ ● likely construction terrain               │
│ Construction score: ████████░░  82%         │
├─────────────────────────────────────────────┤
│ NDVI 2022    -0.08  ● bare/construction     │
│ NDVI 2024    -0.14  ● bare/construction     │
│ Depth mean    0.43                          │
│ Roughness    12.4   ▲ rough                 │
│ Area         1 840 px                       │
│ SAM IoU        91%                          │
├─────────────────────────────────────────────┤
│ TOP CLIP MATCHES                            │
│ ▶ excavation pit earthwork…   34%           │
│ · bare soil gravel sand…      28%           │
│ · construction site cranes…   19%           │
└─────────────────────────────────────────────┘
```

### Download

Every analysis writes `outputs/segments_analysis.gpkg`.
Click **⬇ Download GeoPackage** in the sidebar, or:

```bash
curl http://localhost:8000/api/download/results.gpkg -o segments_analysis.gpkg
```

Load in QGIS: `Layer → Add Vector Layer → segments_analysis.gpkg`

---

## How it works

### Imagery — PDOK WMS

Both RGB and CIR (Color-Infrared) tiles are fetched live for 2022 and 2024.

```
RGB  https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0   layers: 2022_ortho25 / 2024_ortho25
CIR  https://service.pdok.nl/hwh/luchtfotocir/wms/v1_0   layers: 2022_ortho25 / 2024_ortho25
```

> PDOK requires a minimum **600 × 600 m** bbox to return non-blank tiles.
> The server expands the drawn bbox internally; all results are clipped back
> to the **exact drawn area** before returning.

### NDVI

```
NDVI = (NIR − Red) / (NIR + Red + ε)
```

The CIR bands map: Channel 0 = NIR, Channel 1 = Red.
Result is in `[−1, 1]`. Rendered with **RdYlGn** (matplotlib) — red = bare soil / construction, green = vegetation.

### Depth

Depth Anything V2 Small runs on the RGB tile and produces a relative depth map normalised to `[0, 1]`.
Rendered with **inferno** (matplotlib) — dark = near, bright = far.

Per segment, the **depth variance** is used as a roughness signal:
construction sites have disturbed, irregular terrain → high variance.

### Dense SAM2 segmentation

```python
SAM2AutomaticMaskGenerator(
    model    = SAM2_Hiera_L,
    points_per_side          = 32,   # 1024 prompt points → dense coverage
    pred_iou_thresh          = 0.70,
    stability_score_thresh   = 0.80,
    min_mask_region_area     = 100,
)
```

Masks are sorted **descending by area** before being painted into a label raster.
Large masks fill first; small masks overwrite them → every pixel ends up assigned
to its smallest enclosing segment.

### Construction score

For each segment:

```
ndvi_score    = clip((0.20 − mean_ndvi_2024) / 0.50,  0, 1)
roughness     = clip((depth_variance / 1000) / 0.025,  0, 1)
clip_score    = clip((max_constr_clip_sim − 0.08) / 0.22, 0, 1)

construction_score = 0.40 × ndvi_score
                   + 0.35 × roughness
                   + 0.25 × clip_score
```

CLIP construction labels (indices contributing to `clip_score`):

```
0  construction site with cranes and machinery
1  building under construction concrete frame
2  excavation pit earthwork foundation
3  scaffolding on building facade
4  demolition site rubble and debris
5  concrete pour steel structure being built
13 bare soil gravel sand field
```

### Terrain label assignment

```
score ≥ 0.60                          → likely construction terrain
CLIP water   > 0.22                   → water
CLIP building > 0.22                  → roof / building
CLIP paved   > 0.22                   → paved surface
CLIP veg     > 0.22 or NDVI > 0.35   → vegetation
score ≥ 0.35 or CLIP bare > 0.20     → exposed soil / bare ground
otherwise                             → shadow / unknown
```

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `POST /api/analyze-bbox` | — | Main analysis endpoint |
| `GET  /api/bt2022` | — | BT2022 reference polygons (GeoJSON) |
| `GET  /api/health` | — | `{"ready": bool, "device": "cuda:0"}` |
| `GET  /api/download/results.gpkg` | — | Download last result GeoPackage |

### `POST /api/analyze-bbox`

**Request**
```json
{ "bbox_wgs84": [west, south, east, north] }
```

**Response**
```json
{
  "bounds": [[south, west], [north, east]],
  "overlays": {
    "rgb_2022":   "data:image/png;base64,…",
    "cir_2022":   "data:image/png;base64,…",
    "rgb_2024":   "data:image/png;base64,…",
    "cir_2024":   "data:image/png;base64,…",
    "ndvi_2022":  "data:image/png;base64,…",
    "ndvi_2024":  "data:image/png;base64,…",
    "depth_2022": "data:image/png;base64,…",
    "depth_2024": "data:image/png;base64,…"
  },
  "segments": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": { "type": "Polygon", "coordinates": [...] },
        "properties": {
          "terrain_label":      "likely construction terrain",
          "construction_score": 0.82,
          "color":              "#FF4500",
          "mean_ndvi_2022":     -0.08,
          "mean_ndvi_2024":     -0.14,
          "depth_mean":         0.43,
          "depth_roughness":    12.4,
          "area_px":            1840,
          "top3_clip": [
            { "label": "excavation pit earthwork foundation", "score": 0.34 },
            { "label": "bare soil gravel sand field",         "score": 0.28 },
            { "label": "construction site with cranes…",      "score": 0.19 }
          ],
          "predicted_iou":   0.91,
          "stability_score": 0.87
        }
      }
    ]
  },
  "stats": {
    "total": 142,
    "construction": 23,
    "labels": {
      "likely construction terrain": 23,
      "vegetation": 61,
      "paved surface": 28,
      "roof / building": 19,
      "exposed soil / bare ground": 8,
      "shadow / unknown": 3
    }
  }
}
```

---

## Models

| Model | Purpose | Approx VRAM |
|---|---|---|
| [SAM2 Hiera-L](https://huggingface.co/facebook/sam2.1-hiera-large) | Dense auto-segmentation | ~6 GB |
| [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14) | Per-segment semantic classification | ~1.5 GB |
| [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | Depth map → terrain roughness | ~0.3 GB |

All three are downloaded automatically from HuggingFace on first run.
Total VRAM: ~8 GB. Tested on NVIDIA RTX 5070 Laptop.

CPU inference works but is much slower (~5–15 min per bbox).

---

## Requirements

```
Python         3.11+
PyTorch        2.3+  (tested on 2.11 + CUDA 12.8)
CUDA           12.x  (recommended; CPU fallback available)
VRAM           ≥ 8 GB
Disk           ~5 GB for model weights (auto-downloaded)
```

Key packages (see `requirements.txt` for full list):

| Package | Version tested |
|---|---|
| torch | 2.11.0+cu128 |
| transformers | 5.3.0 |
| sam2 | from GitHub |
| fastapi | 0.135.2 |
| geopandas | 1.1.3 |
| rasterio | 1.4.4 |
| shapely | 2.1.2 |
| opencv-python | 4.13.0 |
| numpy | 2.3.5 |
| Pillow | 12.0.0 |
| pyproj | 3.6+ |

---

## Reference data

`data/raw/BT2022.gpkg` — 3 664 construction-site polygons from the Dutch land-use survey (2022), EPSG:28992.

When present, these are shown on the map as an orange dashed reference layer and served via `GET /api/bt2022`.

---

## Project structure

```
construction-change-detector/
├── dashboard/
│   ├── app.py              # FastAPI backend — models, analysis pipeline, API
│   └── static/
│       └── index.html      # Leaflet.js frontend
├── data/
│   ├── pdok_downloader.py  # PDOK WMS client (RGB + CIR, 8 cm / 25 cm)
│   ├── dataset.py
│   └── raw/
│       └── BT2022.gpkg     # Reference construction polygons (place here)
├── outputs/
│   └── segments_analysis.gpkg   # Written after each analysis run
├── requirements.txt
└── README.md
```

---

## Data sources

| | Source | Detail |
|---|---|---|
| RGB imagery | [PDOK Luchtfoto RGB WMS](https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0) | 8 cm orthoHR · layers `2022_ortho25` / `2024_ortho25` |
| CIR imagery | [PDOK Luchtfoto CIR WMS](https://service.pdok.nl/hwh/luchtfotocir/wms/v1_0) | Color-infrared · layers `2022_ortho25` / `2024_ortho25` |
| Reference polygons | BT2022.gpkg | Dutch CBS land-use survey, 3 664 active construction sites |

<details>
<summary>PDOK WMS known gotchas</summary>

| Issue | Symptom | Fix |
|---|---|---|
| Small bbox | Returns pure-white blank tile | Enforce 600 × 600 m minimum; server does this automatically |
| `TIME` parameter | Silently ignored | Use year-specific layer names: `2022_ortho25` |
| `owslib` inconsistencies | Random failures | Use `requests.get` directly |
| 8 cm blank tile | Some rural areas only have 25 cm coverage | Server auto-falls back to `25cm` resolution |

</details>

---

<div align="center">
<i>Named after Sam Spade — the detective who always finds what's hidden.</i>
</div>
