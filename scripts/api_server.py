"""
Hydrospatial Data Cube API Server.

Loads layers and config from yaml. Serves:
  GET /                       viewer HTML
  GET /api/layers             tile URLs + map bounds + feature flags
  GET /api/query              point query across all NetCDF variables
  GET /api/stats              region stats (EPSG:3826)
  GET /api/profile            depth/sediment/rl along polyline
  GET /api/3d-scene           heightmap + SSS texture for 3D block view
  GET /api/tracklines         survey tracklines GeoJSON
  GET /api/waterfall-index    waterfall image index for SSS/SBP viewer
"""
import json
import math
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import rasterio
import uvicorn
import xarray as xr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from localtileserver import TileClient
from pyproj import Transformer
from rasterio.windows import from_bounds

from src.config import ROOT, get_config
from src.sbp.config import SEDIMENT_LABELS

cfg = get_config()
viewer_cfg = cfg["viewer"]

NC_PATH = ROOT / viewer_cfg["netcdf"]
TRACKLINES_PATH = ROOT / viewer_cfg["tracklines"]
WATERFALLS_DIR = ROOT / viewer_cfg["waterfalls_dir"]
VIEWER_DIR = ROOT / viewer_cfg["static_dir"]
SSS_HF_TIF = next(
    (ROOT / l["path"] for l in viewer_cfg["tile_layers"] if l["id"] == "imagery_hf"),
    None,
)

EPSG = cfg["grid"]["epsg"]
from_ll = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)
to_ll = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)


def percentile_range(tif_path, low=2, high=98):
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        valid = data[np.isfinite(data)]
    if not len(valid):
        return None, None
    return float(np.percentile(valid, low)), float(np.percentile(valid, high))


# Module-level state, populated in lifespan startup
state = {
    "ds": None,
    "tile_clients": {},
    "center": [22.137, 120.785],
    "bounds_ll": None,
    "has_isopach": False,
}


@asynccontextmanager
async def lifespan(_app):
    if not NC_PATH.exists():
        raise FileNotFoundError(
            f"Data cube not found: {NC_PATH}. Run build_datacube.py first."
        )
    state["ds"] = xr.open_dataset(NC_PATH)
    state["has_isopach"] = "isopach" in state["ds"].data_vars
    print(f"Cube loaded: {list(state['ds'].data_vars)}")
    print(f"  isopach available: {state['has_isopach']}")

    for layer in viewer_cfg["tile_layers"]:
        tif_path = ROOT / layer["path"]
        if not tif_path.exists():
            print(f"  Skip tile: {tif_path.relative_to(ROOT)} not found")
            continue

        client = TileClient(str(tif_path))
        url_kwargs = {}

        if layer.get("palette"):
            url_kwargs["colormap"] = layer["palette"]
            vmin, vmax = percentile_range(tif_path)
            if vmin is not None:
                url_kwargs["vmin"] = vmin
                url_kwargs["vmax"] = vmax

        nodata = client.dataset.nodata
        if nodata is not None:
            url_kwargs["nodata"] = nodata

        state["tile_clients"][layer["id"]] = {
            "client": client,
            "url": client.get_tile_url(**url_kwargs),
            "label": layer["label"],
            "center": client.center(),
            "bounds": client.bounds(),
        }
        print(f"  Tile: {layer['id']} ready")

    if "bathymetry" in state["tile_clients"]:
        info = state["tile_clients"]["bathymetry"]
        state["center"] = list(info["center"])
        b = info["bounds"]
        state["bounds_ll"] = [[b[0], b[2]], [b[1], b[3]]]

    yield

    # Cleanup
    if state["ds"] is not None:
        state["ds"].close()


app = FastAPI(title="Hydrospatial Data Cube", lifespan=lifespan)
VIEWER_DIR.mkdir(exist_ok=True)
app.mount("/viewer", StaticFiles(directory=VIEWER_DIR), name="viewer")
if WATERFALLS_DIR.exists():
    app.mount("/waterfalls", StaticFiles(directory=WATERFALLS_DIR), name="waterfalls")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = VIEWER_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>viewer/index.html not found</h1>"


@app.get("/api/layers")
async def get_layers():
    return {
        "layers": {k: {"label": v["label"], "url": v["url"]}
                   for k, v in state["tile_clients"].items()},
        "center": state["center"],
        "bounds": state["bounds_ll"],
        "features": {
            "has_isopach": state["has_isopach"],
        },
    }


@app.get("/api/query")
async def query_point(lat: float, lon: float):
    ds = state["ds"]
    x, y = from_ll.transform(lon, lat)

    if (x < ds.x.values.min() or x > ds.x.values.max() or
            y < ds.y.values.min() or y > ds.y.values.max()):
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

        if val is None or np.isnan(val):
            result[var] = {"name": long_name, "value": None, "units": units}
        elif var == "sediment_class":
            idx = int(val)
            label = SEDIMENT_LABELS[idx] if 0 <= idx < len(SEDIMENT_LABELS) else f"class {idx}"
            result[var] = {"name": long_name, "value": label, "units": "", "class_id": idx}
        else:
            result[var] = {"name": long_name, "value": round(val, 4), "units": units}

    return result


@app.get("/api/stats")
async def region_stats(x0: float, y0: float, x1: float, y1: float):
    ds = state["ds"]
    region = ds.sel(
        x=slice(min(x0, x1), max(x0, x1)),
        y=slice(max(y0, y1), min(y0, y1)),
    )

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

        if not len(valid):
            result["layers"][var] = {"name": long_name, "value": None}
            continue

        if var == "sediment_class":
            classes, counts = np.unique(valid.astype(int), return_counts=True)
            dom = int(classes[np.argmax(counts)])
            label = SEDIMENT_LABELS[dom] if 0 <= dom < len(SEDIMENT_LABELS) else f"class {dom}"
            result["layers"][var] = {
                "name": long_name,
                "dominant": label,
                "purity": round(100 * counts.max() / counts.sum(), 1),
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


@app.get("/api/profile")
async def profile(coords: str):
    """Depth, sediment_class, rl (and isopach if available) along a polyline."""
    ds = state["ds"]
    points = coords.split(";")

    keys = [("bathymetry", "depth"), ("sediment_class", "sediment"), ("rl", "rl")]
    if state["has_isopach"]:
        keys.append(("isopach", "isopach"))

    result = {key: [] for _, key in keys}

    for pt in points:
        try:
            lon, lat = map(float, pt.split(","))
            x, y = from_ll.transform(lon, lat)
            xi = int(np.argmin(np.abs(ds.x.values - x)))
            yi = int(np.argmin(np.abs(ds.y.values - y)))

            for var, key in keys:
                if var in ds.data_vars:
                    val = float(ds[var].values[yi, xi])
                    result[key].append(round(val, 4) if not np.isnan(val) else None)
                else:
                    result[key].append(None)
        except Exception:
            for _, key in keys:
                result[key].append(None)

    return result


@app.get("/api/3d-scene")
async def get_3d_scene(x0: float, y0: float, x1: float, y1: float):
    ds = state["ds"]
    min_x, max_x = min(x0, x1), max(x0, x1)
    min_y, max_y = min(y0, y1), max(y0, y1)

    region = ds.sel(x=slice(min_x, max_x), y=slice(max_y, min_y))
    if region.sizes["x"] == 0 or region.sizes["y"] == 0:
        return {"error": "No data in selected region"}

    max_grid = 150
    step = max(1, math.ceil(max(region.sizes["x"], region.sizes["y"]) / max_grid))
    region_ds = region.isel(x=slice(None, None, step), y=slice(None, None, step))
    w, h = region_ds.sizes["x"], region_ds.sizes["y"]

    bathy = region_ds["bathymetry"].fillna(region_ds["bathymetry"].mean())
    bathymetry_1d = bathy.values.flatten().tolist()

    bedrock_1d = None
    if state["has_isopach"] and "isopach" in region_ds.data_vars:
        iso = region_ds["isopach"].fillna(0)
        bedrock_1d = (bathy + iso).values.flatten().tolist()

    sss_texture_1d = None
    if SSS_HF_TIF and SSS_HF_TIF.exists():
        try:
            with rasterio.open(SSS_HF_TIF) as src:
                window = from_bounds(min_x, min_y, max_x, max_y, transform=src.transform)
                sss_arr = src.read(1, window=window, out_shape=(h, w))
                if src.nodata is not None:
                    sss_arr[sss_arr == src.nodata] = 0
                lo, hi = np.nanmin(sss_arr), np.nanmax(sss_arr)
                if hi > lo:
                    norm = ((sss_arr - lo) / (hi - lo) * 255).astype(np.uint8)
                    sss_texture_1d = norm[::-1, :].flatten().tolist()
                else:
                    sss_texture_1d = [0] * (w * h)
        except Exception as e:
            print(f"SSS texture read failed: {e}")

    return {
        "width": w,
        "height": h,
        "step_m": cfg["grid"]["resolution"] * step,
        "bathymetry": bathymetry_1d,
        "bedrock": bedrock_1d,
        "sss_texture": sss_texture_1d,
    }


@app.get("/api/tracklines")
async def get_tracklines():
    if TRACKLINES_PATH.exists():
        return JSONResponse(json.loads(TRACKLINES_PATH.read_text()))
    return {"type": "FeatureCollection", "features": []}


@app.get("/api/waterfall-index")
async def get_waterfall_index():
    index_path = WATERFALLS_DIR / "index.json"
    if index_path.exists():
        return JSONResponse(json.loads(index_path.read_text()))
    return {"sss": {}, "sbp": {}}


if __name__ == "__main__":
    print(f"\nViewer: http://localhost:8000")
    print(f"API:    http://localhost:8000/api/layers")
    uvicorn.run(app, host="0.0.0.0", port=8000)