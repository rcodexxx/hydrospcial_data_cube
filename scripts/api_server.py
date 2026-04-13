# scripts/api_server.py
"""
Hydrospatial Data Cube API Server

Usage:
    python scripts/api_server.py

Endpoints:
    GET /                     → viewer HTML page
    GET /api/query?lat=&lon=  → query all layers at a point
    GET /api/stats?x0=&y0=&x1=&y1= → region statistics (EPSG:3826)
    GET /api/layers            → available tile layers
    GET /api/tracklines        → survey tracklines GeoJSON
"""

import json
from pathlib import Path

import numpy as np
import uvicorn
import xarray as xr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from localtileserver import TileClient
from pyproj import Transformer

# ── Config ───────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
NC_PATH = ROOT / "outputs" / "hydrospatial_datacube.nc"
TIF_DIR = ROOT / "outputs" / "tif"
VIEWER_DIR = ROOT / "src" / "viewer"
TRACKLINES_PATH = ROOT / "outputs" / "tracklines.json"

from_ll = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True)
to_ll = Transformer.from_crs("EPSG:3826", "EPSG:4326", always_xy=True)

SEDIMENT_LABELS = [
    "Coarse sand",
    "Fine sand",
    "Very fine sand",
    "Silty sand",
    "Sandy silt",
    "Silt",
    "Sandy-silt-clay",
    "Silty clay",
    "Clayey silt",
    "Framework-supported mud",
    "Fluid mud",
]

# ── Layer definitions ────────────────────────────────────────
TILE_LAYERS = {
    "bathymetry": {
        "tif": "mbes_bathymetry.tif", 
        "label": "Bathymetry",
        "palette": "turbo_r"
    },
    "imagery_lf": {
        "tif": "sss_imagery_lf.tif", 
        "label": "SSS Imagery LF",
        "palette": "copper"
    },
    "imagery_hf": {
        "tif": "sss_imagery_hf.tif", 
        "label": "SSS Imagery HF",
        "palette": "copper"
    },
        "sediment_class": {
        "tif": "sbp_sediment_rgb.tif", 
        "label": "Sediment Class",
        "palette": None 
    },
    "mag_residual": {
        "tif": "mag_residual.tif", 
        "label": "Magnetic Residual",
        "palette": "rdbu"
    },
}

# ── Load NC ──────────────────────────────────────────────────
ds = xr.open_dataset(NC_PATH)
print(f"NC loaded: {list(ds.data_vars)}")
print(f"Grid: x={len(ds.x)}, y={len(ds.y)}")

# ── Start tile servers ───────────────────────────────────────
tile_clients = {}

for key, cfg in TILE_LAYERS.items():
    tif_path = TIF_DIR / cfg["tif"]
    if tif_path.exists():
        client = TileClient(str(tif_path))
        url_kwargs = {}
        
        if cfg.get("palette"):
            url_kwargs["colormap"] = cfg["palette"]
        else:
            url_kwargs["vmin"] = 0
            url_kwargs["vmax"] = 255
        
        native_nodata = client.dataset.nodata
        if native_nodata is not None:
            url_kwargs["nodata"] = native_nodata

        colored_url = client.get_tile_url(**url_kwargs)
        
        tile_clients[key] = {
            "client": client,
            "url": colored_url, 
            "label": cfg["label"],
            "center": client.center(),
            "bounds": client.bounds(),
        }
        print(f"  Tile: {key} → {colored_url}")
    else:
        print(f"  Skip: {tif_path} not found")

# get center from bathymetry
center = [22.137, 120.785]
bounds_ll = None
if "bathymetry" in tile_clients:
    center = list(tile_clients["bathymetry"]["center"])
    b = tile_clients["bathymetry"]["bounds"]
    bounds_ll = [[b[0], b[2]], [b[1], b[3]]]  # [[south, west], [north, east]]

# ── FastAPI app ──────────────────────────────────────────────
app = FastAPI(title="Hydrospatial Data Cube")

# serve viewer HTML
VIEWER_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=VIEWER_DIR), name="static")
app.mount("/waterfalls", StaticFiles(directory=ROOT / "outputs" / "waterfalls"), name="waterfalls")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = VIEWER_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>viewer/index.html not found</h1>"


@app.get("/api/layers")
async def get_layers():
    layers = {}
    for key, info in tile_clients.items():
        layers[key] = {
            "label": info["label"],
            "url": info["url"],
        }
    return {
        "layers": layers,
        "center": center,
        "bounds": bounds_ll,
    }


@app.get("/api/query")
async def query_point(lat: float, lon: float):
    x, y = from_ll.transform(lon, lat)

    if (
        x < ds.x.values.min()
        or x > ds.x.values.max()
        or y < ds.y.values.min()
        or y > ds.y.values.max()
    ):
        return {"error": "Outside data bounds"}

    xi = int(np.argmin(np.abs(ds.x.values - x)))
    yi = int(np.argmin(np.abs(ds.y.values - y)))

    result = {"lat": lat, "lon": lon, "x_3826": round(x, 1), "y_3826": round(y, 1)}

    for var in ds.data_vars:
        try:
            val = float(ds[var].values[yi, xi])
        except (IndexError, ValueError):
            val = None

        long_name = ds[var].attrs.get("long_name", var)
        units = ds[var].attrs.get("units", "")

        if np.isnan(val) if val is not None else True:
            result[var] = {"name": long_name, "value": None, "units": units}
        elif var == "sediment_class":
            idx = int(val)
            label = (
                SEDIMENT_LABELS[idx]
                if 0 <= idx < len(SEDIMENT_LABELS)
                else f"class {idx}"
            )
            result[var] = {
                "name": long_name,
                "value": label,
                "units": "",
                "class_id": idx,
            }
        else:
            result[var] = {"name": long_name, "value": round(val, 4), "units": units}

    return result


@app.get("/api/stats")
async def region_stats(x0: float, y0: float, x1: float, y1: float):
    """Region stats. Expects EPSG:3826 coordinates."""
    region = ds.sel(
        x=slice(min(x0, x1), max(x0, x1)),
        y=slice(max(y0, y1), min(y0, y1)),
    )

    # convert corners to lat/lon
    lon0, lat0 = to_ll.transform(x0, y0)
    lon1, lat1 = to_ll.transform(x1, y1)

    result = {
        "width_m": round(abs(x1 - x0), 1),
        "height_m": round(abs(y1 - y0), 1),
        "sw": {"lat": round(min(lat0, lat1), 6), "lon": round(min(lon0, lon1), 6)},
        "ne": {"lat": round(max(lat0, lat1), 6), "lon": round(max(lon0, lon1), 6)},
        "layers": {},
    }

    for var in ds.data_vars:
        vals = region[var].values.ravel()
        valid = vals[np.isfinite(vals)]
        long_name = ds[var].attrs.get("long_name", var)
        units = ds[var].attrs.get("units", "")

        if len(valid) == 0:
            result["layers"][var] = {"name": long_name, "value": None}
            continue

        if var == "sediment_class":
            classes, counts = np.unique(valid.astype(int), return_counts=True)
            dominant = int(classes[np.argmax(counts)])
            label = (
                SEDIMENT_LABELS[dominant]
                if 0 <= dominant < len(SEDIMENT_LABELS)
                else f"class {dominant}"
            )
            pct = round(100 * counts.max() / counts.sum(), 1)
            result["layers"][var] = {
                "name": long_name,
                "dominant": label,
                "purity": pct,
            }
        else:
            result["layers"][var] = {
                "name": long_name,
                "units": units,
                "min": round(float(valid.min()), 4),
                "max": round(float(valid.max()), 4),
                "mean": round(float(valid.mean()), 4),
            }

    return result


@app.get("/api/tracklines")
async def get_tracklines():
    if TRACKLINES_PATH.exists():
        return JSONResponse(json.loads(TRACKLINES_PATH.read_text()))
    return {"type": "FeatureCollection", "features": []}


@app.get("/api/waterfall-index")
async def get_waterfall_index():
    index_path = ROOT / "outputs" / "waterfalls" / "index.json"
    if index_path.exists():
        return JSONResponse(json.loads(index_path.read_text()))
    return {"sss": {}, "sbp": {}}

@app.get("/api/depth-profile")
async def depth_profile(coords: str):
    """Get depth along a series of coordinates.
    coords format: lon1,lat1;lon2,lat2;...
    """
    points = coords.split(";")
    depths = []
    for pt in points:
        try:
            lon, lat = map(float, pt.split(","))
            x, y = from_ll.transform(lon, lat)
            xi = int(np.argmin(np.abs(ds.x.values - x)))
            yi = int(np.argmin(np.abs(ds.y.values - y)))
            val = float(ds["bathymetry"].values[yi, xi])
            depths.append(round(val, 2) if not np.isnan(val) else None)
        except:
            depths.append(None)
    return {"depths": depths}


@app.get("/api/profile")
async def profile(coords: str):
    """Get depth, sediment, isopach along a series of coordinates.
    coords format: lon1,lat1;lon2,lat2;...
    """
    points = coords.split(";")
    result = {"depth": [], "sediment": [], "isopach": [], "rl": []}
    
    for pt in points:
        try:
            lon, lat = map(float, pt.split(","))
            x, y = from_ll.transform(lon, lat)
            xi = int(np.argmin(np.abs(ds.x.values - x)))
            yi = int(np.argmin(np.abs(ds.y.values - y)))
            
            for var, key in [("bathymetry", "depth"), 
                            ("sediment_class", "sediment"),
                            ("isopach", "isopach"), 
                            ("rl", "rl")]:
                if var in ds.data_vars:
                    val = float(ds[var].values[yi, xi])
                    result[key].append(round(val, 4) if not np.isnan(val) else None)
                else:
                    result[key].append(None)
        except:
            for key in result:
                result[key].append(None)
    
    return result


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nViewer: http://localhost:8000")
    print(f"API:    http://localhost:8000/api/layers")
    uvicorn.run(app, host="0.0.0.0", port=8000)
