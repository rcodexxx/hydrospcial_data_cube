"""
Generate bathymetric contour GeoJSON from MBES DEM.

Output is consumed by the viewer (loaded as static asset). Uses the
same Gaussian smoothing and isobath levels as plot_sub_bottom.py to
keep the viewer visually consistent with thesis figures.

Coordinates are converted to EPSG:4326 for direct Leaflet consumption.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

from src.config import ROOT, get_config


# Match plot_sub_bottom.py constants
ISOBATH_MAJOR_M = 10
ISOBATH_MINOR_M = 5
DEM_SMOOTH_SIGMA_PX = 10


def smooth_dem(dem, sigma_px=DEM_SMOOTH_SIGMA_PX):
    mask = np.isfinite(dem)
    filled = np.where(mask, dem, 0)
    weights = mask.astype(np.float32)
    sm = gaussian_filter(filled, sigma=sigma_px)
    w_sm = gaussian_filter(weights, sigma=sigma_px)
    return np.where(w_sm > 0.1, sm / np.maximum(w_sm, 1e-10), np.nan)


def contour_to_geojson(dem_sm, levels, transform, src_crs, dst_crs="EPSG:4326"):
    """Compute contours via matplotlib, convert pixel coords to lon/lat."""
    tr = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    # matplotlib contour returns paths in pixel space (col, row) order
    fig, ax = plt.subplots()
    cs = ax.contour(dem_sm, levels=levels)
    plt.close(fig)

    features = []
    for level, allsegs in zip(cs.levels, cs.allsegs):
        is_major = (int(level) % ISOBATH_MAJOR_M == 0)
        for seg in allsegs:
            if len(seg) < 2:
                continue
            cols, rows = seg[:, 0], seg[:, 1]
            xs = transform.c + (cols + 0.5) * transform.a
            ys = transform.f + (rows + 0.5) * transform.e
            lons, lats = tr.transform(xs, ys)
            coords = [[float(lon), float(lat)] for lon, lat in zip(lons, lats)]
            features.append({
                "type": "Feature",
                "properties": {
                    "depth_m": float(level),
                    "level": "major" if is_major else "minor",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
            })
    return features


def main():
    cfg = get_config()
    mbes_path = ROOT / cfg["mbes"]["bathymetry_tif"]
    out_path = ROOT / cfg["viewer"]["contours"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(mbes_path) as src:
        dem = src.read(1).astype(np.float32)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        transform = src.transform
        src_crs = src.crs

    print("Smoothing DEM...")
    dem_sm = smooth_dem(dem)
    depth_max = int(np.ceil(np.nanmax(dem_sm) / ISOBATH_MAJOR_M) * ISOBATH_MAJOR_M)

    minor = [d for d in range(ISOBATH_MINOR_M, depth_max, ISOBATH_MINOR_M)
             if d % ISOBATH_MAJOR_M != 0]
    major = list(range(ISOBATH_MAJOR_M, depth_max, ISOBATH_MAJOR_M))
    levels = sorted(set(minor + major))

    print(f"Computing contours: {len(levels)} levels ({len(major)} major, {len(minor)} minor)")
    features = contour_to_geojson(dem_sm, levels, transform, src_crs)

    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w") as f:
        json.dump(geojson, f)

    n_major = sum(1 for f in features if f["properties"]["level"] == "major")
    n_minor = sum(1 for f in features if f["properties"]["level"] == "minor")
    print(f"\nSaved: {out_path.relative_to(ROOT)}")
    print(f"  segments: {len(features)} ({n_major} major, {n_minor} minor)")


if __name__ == "__main__":
    main()