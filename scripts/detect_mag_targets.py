"""
Detect magnetic targets from anomaly grid.

Identifies local |anomaly| extrema above threshold, clusters nearby
extrema (within DEDUP_RADIUS_M) to avoid double-counting dipole pairs,
and writes both a CSV and a GeoJSON target list for downstream
consumption (CSV for analysis, GeoJSON for the viewer's Leaflet layer).

The output is intentionally minimal — coordinates, anomaly amplitude,
polarity. Visualization (markers, labels, classification) happens in
the UI layer, not here.
"""
import csv
import json

import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import maximum_filter

from src.config import get_config, ROOT
from src.mag.config import (
    TARGET_THRESHOLD_NT,
    TARGET_PEAK_FOOTPRINT_PX,
    TARGET_DEDUP_RADIUS_M,
)


def find_local_extrema(arr, threshold, footprint_px):
    """
    Locate local maxima of |arr| within `footprint_px` neighborhoods
    where |arr| > threshold.
    Returns (rows, cols, signed_values).
    """
    abs_arr = np.where(np.isfinite(arr), np.abs(arr), 0)
    local_max = maximum_filter(abs_arr, size=footprint_px) == abs_arr
    above = abs_arr > threshold
    mask = local_max & above & np.isfinite(arr)
    rows, cols = np.where(mask)
    return rows, cols, arr[rows, cols]


def dedup_targets(xs, ys, vals, radius_m):
    """
    Greedy: sort by |val| descending, accept point if no stronger
    target within radius_m already accepted.
    """
    order = np.argsort(-np.abs(vals))
    keep_x, keep_y, keep_v = [], [], []
    for i in order:
        x, y, v = xs[i], ys[i], vals[i]
        too_close = False
        for kx, ky in zip(keep_x, keep_y):
            if (x - kx) ** 2 + (y - ky) ** 2 < radius_m ** 2:
                too_close = True
                break
        if not too_close:
            keep_x.append(x)
            keep_y.append(y)
            keep_v.append(v)
    return np.array(keep_x), np.array(keep_y), np.array(keep_v)


def main():
    cfg = get_config()
    epsg = cfg["grid"]["epsg"]
    an_tif = ROOT / cfg["mag"]["outputs"]["anomaly_tif"]
    out_csv = ROOT / cfg["mag"]["outputs"]["targets_csv"]
    out_geojson = out_csv.with_suffix(".geojson")

    print(f"Loading anomaly grid: {an_tif.name}")
    with rasterio.open(an_tif) as src:
        arr = src.read(1).astype(np.float32)
        nd = src.nodata
        transform = src.transform
        res = transform.a
    if nd is not None:
        arr[arr == nd] = np.nan

    footprint_px = max(int(round(TARGET_PEAK_FOOTPRINT_PX)), 3)
    print(f"Searching local extrema: |a| > {TARGET_THRESHOLD_NT:.0f} nT, "
          f"footprint {footprint_px}×{footprint_px} px")
    rows, cols, vals = find_local_extrema(
        arr, TARGET_THRESHOLD_NT, footprint_px
    )
    print(f"  Raw extrema    : {len(rows)}")

    # Pixel → projected coords
    xs = transform.c + (cols + 0.5) * res
    ys = transform.f + (rows + 0.5) * (-res)

    # Dedup nearby (dipole pairs, peak shoulders)
    xs, ys, vals = dedup_targets(xs, ys, vals, TARGET_DEDUP_RADIUS_M)
    print(f"  After dedup ({TARGET_DEDUP_RADIUS_M:.0f} m radius) : "
          f"{len(xs)}")

    # Project → lon/lat
    transformer = Transformer.from_crs(
        f"EPSG:{epsg}", "EPSG:4326", always_xy=True
    )
    lons, lats = transformer.transform(xs, ys)

    # Sort by |val| descending for stable target IDs
    order = np.argsort(-np.abs(vals))
    xs, ys, vals = xs[order], ys[order], vals[order]
    lons, lats = lons[order], lats[order]

    # Build targets list once, write to both formats
    targets = [
        {
            "target_id": f"T{i:03d}",
            "x_m": float(x),
            "y_m": float(y),
            "lon": float(lon),
            "lat": float(lat),
            "anomaly_nT": float(v),
            "polarity": "+" if v > 0 else "-",
        }
        for i, (x, y, lon, lat, v) in enumerate(
            zip(xs, ys, lons, lats, vals), start=1
        )
    ]

    # CSV (existing format, backward compatible)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target_id", "x_m", "y_m", "lon", "lat",
                         "anomaly_nT", "polarity"])
        for t in targets:
            writer.writerow([
                t["target_id"],
                f"{t['x_m']:.2f}", f"{t['y_m']:.2f}",
                f"{t['lon']:.6f}", f"{t['lat']:.6f}",
                f"{t['anomaly_nT']:+.1f}",
                t["polarity"],
            ])

    # GeoJSON for viewer (Leaflet GeoJSON layer)
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [t["lon"], t["lat"]],
                },
                "properties": {
                    "target_id": t["target_id"],
                    "anomaly_nT": round(t["anomaly_nT"], 1),
                    "polarity": t["polarity"],
                    "x_m": round(t["x_m"], 2),
                    "y_m": round(t["y_m"], 2),
                },
            }
            for t in targets
        ],
    }
    with open(out_geojson, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    print(f"\nSaved: {out_csv.name}  ({len(xs)} targets)")
    print(f"Saved: {out_geojson.name}")
    if len(vals):
        print(f"Strongest        : {vals[np.argmax(np.abs(vals))]:+.1f} nT")
        print(f"|anomaly| range  : {np.abs(vals).min():.0f} ~ "
              f"{np.abs(vals).max():.0f} nT")


if __name__ == "__main__":
    main()