# scripts/build/build_mag.py
"""
Build magnetometer layers: background field and residual anomaly.

Workflow:
  1. Read .mag files, filter anomalous F values and low quality records
  2. Apply IGRF correction per survey date
  3. Along-track median filter: large window (background), small window (anomaly)
  4. IDW interpolation to MBES grid within MAX_EXTRAP_M of tracks
  5. Gaussian low-pass on background field
  6. Residual = anomaly - background

Outputs:
  mag_background.tif  - Large-scale geological magnetic trend (nT)
  mag_residual.tif    - Local magnetic anomaly for UCH detection (nT)
  mag_confidence.tif  - 0=measured, 1=interpolated, 255=nodata
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import ppigrf
import rasterio
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from scipy.spatial import KDTree
from tqdm import tqdm

from src.data_loader.read_mag import read_mag
from src.config import EPSG

ROOT     = Path(__file__).parent.parent.parent
MAG_DIRS = {
    ROOT / "data/mag/20251223": datetime(2025, 12, 23),
    ROOT / "data/mag/20260108": datetime(2026, 1,  8),
}
MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
OUT_BG   = ROOT / "outputs/tif/mag_background.tif"
OUT_RES  = ROOT / "outputs/tif/mag_residual.tif"
OUT_CONF = ROOT / "outputs/tif/mag_confidence.tif"

LAT_C = 22.137
LON_C = 120.787

MAX_EXTRAP_M  = 50.0
IDW_POWER     = 2
IDW_NEIGHBORS = 12

# record spacing ~2.94m
# WIN_BG  ~100m → removes local anomalies, keeps regional background
# WIN_ANOM ~15m → removes electronic noise, preserves UCH anomalies
WIN_BG   = 35
WIN_ANOM = 5

SMOOTH_M      = 200.0
QUALITY_MIN   = 90
F_TOLERANCE   = 0.10   # accept F within ±10% of IGRF


def get_igrf(date):
    Be, Bn, Bu = ppigrf.igrf(LON_C, LAT_C, 0.0, date)
    return np.sqrt(np.array(Be)**2 + np.array(Bn)**2 + np.array(Bu)**2).item()


def idw(query_pts, known_pts, known_vals, k, power):
    tree = KDTree(known_pts)
    dists, idxs = tree.query(query_pts, k=k)
    dists = np.maximum(dists, 1e-6)
    w = 1.0 / dists ** power
    w /= w.sum(axis=1, keepdims=True)
    return (w * known_vals[idxs]).sum(axis=1)


def load_and_filter(mag_dirs):
    bg_x,   bg_y,   bg_v   = [], [], []
    anom_x, anom_y, anom_v = [], [], []
    total_raw = total_kept = 0

    for mag_dir, date in mag_dirs.items():
        F_igrf = get_igrf(date)
        F_lo   = F_igrf * (1 - F_TOLERANCE)
        F_hi   = F_igrf * (1 + F_TOLERANCE)
        print(f"\n{mag_dir.name}: IGRF={F_igrf:.1f} nT  "
              f"valid F range=[{F_lo:.0f}, {F_hi:.0f}]")

        for mag_file in tqdm(sorted(mag_dir.glob("*.mag")),
                             desc=f"  {mag_dir.name}"):
            records = read_mag(mag_file, apply_layback=True)
            if len(records) < WIN_BG:
                continue

            xs  = np.array([r["x"]       for r in records])
            ys  = np.array([r["y"]       for r in records])
            F   = np.array([r["F_nT"]    for r in records])
            q   = np.array([r["quality"] for r in records])
            total_raw += len(records)

            # quality filter
            ok = q >= QUALITY_MIN
            # F value filter: reject physically impossible values
            ok &= (F >= F_lo) & (F <= F_hi)

            if ok.sum() < WIN_BG:
                continue

            xs, ys, F = xs[ok], ys[ok], F[ok]
            total_kept += len(xs)

            anom_raw = F - F_igrf

            # background: large window removes local anomalies
            win_bg = min(WIN_BG, len(anom_raw))
            if win_bg % 2 == 0:
                win_bg -= 1
            anom_bg = medfilt(anom_raw, kernel_size=win_bg)

            # anomaly: small window removes electronic noise
            win_an = min(WIN_ANOM, len(anom_raw))
            if win_an % 2 == 0:
                win_an -= 1
            anom_an = medfilt(anom_raw, kernel_size=win_an)

            bg_x.extend(xs);   bg_y.extend(ys);   bg_v.extend(anom_bg)
            anom_x.extend(xs); anom_y.extend(ys); anom_v.extend(anom_an)

    print(f"\nTotal records: {total_raw} raw → {total_kept} kept "
          f"({100*total_kept/max(total_raw,1):.1f}%)")

    return (np.array(bg_x),   np.array(bg_y),   np.array(bg_v),
            np.array(anom_x), np.array(anom_y), np.array(anom_v))


def main():
    print("Loading and filtering MAG data...")
    bg_x, bg_y, bg_v, anom_x, anom_y, anom_v = load_and_filter(MAG_DIRS)

    # remove residual outliers from background (>2σ)
    bg_std   = bg_v.std()
    bg_clean = np.abs(bg_v) < 2.0 * bg_std
    bg_x, bg_y, bg_v = bg_x[bg_clean], bg_y[bg_clean], bg_v[bg_clean]
    bg_pts   = np.column_stack([bg_x,   bg_y])
    anom_pts = np.column_stack([anom_x, anom_y])

    print(f"Background pts : {len(bg_x)} "
          f"(removed {(~bg_clean).sum()} outliers)  "
          f"std={bg_v.std():.1f} nT")
    print(f"Anomaly pts    : {len(anom_x)}  std={anom_v.std():.1f} nT")

    # load MBES grid
    with rasterio.open(MBES_TIF) as src:
        profile     = src.profile.copy()
        transform   = src.transform
        height, width = src.height, src.width
        mbes_data   = src.read(1)
        mbes_nodata = src.nodata

    res     = transform.a
    xs_grid = transform.c + (np.arange(width)  + 0.5) * res
    ys_grid = transform.f + (np.arange(height) + 0.5) * (-res)
    gx, gy  = np.meshgrid(xs_grid, ys_grid)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

    valid_grid = (
        (mbes_data != mbes_nodata).ravel()
        if mbes_nodata is not None
        else np.isfinite(mbes_data).ravel()
    )

    # distance mask: only interpolate within MAX_EXTRAP_M of tracks
    print(f"\nComputing distance mask (max {MAX_EXTRAP_M} m)...")
    bg_tree  = KDTree(bg_pts)
    dist, _  = bg_tree.query(grid_pts[valid_grid], k=1)
    within   = dist <= MAX_EXTRAP_M
    vg_idx   = np.where(valid_grid)[0]
    valid_grid[vg_idx[~within]] = False
    print(f"Valid grid pts : {valid_grid.sum()} "
          f"({100*valid_grid.sum()/np.sum(mbes_data != mbes_nodata):.1f}% of MBES coverage)")

    valid_indices = np.where(valid_grid)[0]
    chunk = 50000

    # interpolate background
    print("\nInterpolating background field...")
    bg_grid = np.full(len(grid_pts), np.nan)
    for i in tqdm(range(0, len(valid_indices), chunk), desc="IDW background"):
        idx = valid_indices[i:i + chunk]
        bg_grid[idx] = idw(grid_pts[idx], bg_pts, bg_v,
                           k=IDW_NEIGHBORS, power=IDW_POWER)
    bg_2d = bg_grid.reshape(height, width).astype(np.float32)

    # Gaussian low-pass on background
    sigma_px = SMOOTH_M / res
    print(f"Gaussian filter (sigma={sigma_px:.0f} px = {SMOOTH_M:.0f} m)...")
    bg_valid  = np.isfinite(bg_2d)
    bg_filled = np.where(bg_valid, bg_2d, 0.0)
    bg_smooth = gaussian_filter(bg_filled, sigma=sigma_px)
    weight    = gaussian_filter(bg_valid.astype(float), sigma=sigma_px)
    bg_smooth = np.where(weight > 0.1, bg_smooth / weight, np.nan)
    bg_smooth = np.where(bg_valid, bg_smooth, np.nan).astype(np.float32)

    # interpolate anomaly
    print("\nInterpolating anomaly field...")
    anom_grid = np.full(len(grid_pts), np.nan)
    for i in tqdm(range(0, len(valid_indices), chunk), desc="IDW anomaly"):
        idx = valid_indices[i:i + chunk]
        anom_grid[idx] = idw(grid_pts[idx], anom_pts, anom_v,
                             k=IDW_NEIGHBORS, power=IDW_POWER)
    anom_2d = anom_grid.reshape(height, width).astype(np.float32)

    # residual = anomaly - smoothed background
    residual_2d = np.where(
        np.isfinite(anom_2d) & np.isfinite(bg_smooth),
        anom_2d - bg_smooth, np.nan
    ).astype(np.float32)

    # confidence mask
    print("\nBuilding confidence mask...")
    conf_flat  = np.full(len(grid_pts), 255, dtype=np.uint8)
    an_tree    = KDTree(anom_pts)
    dist_v, _  = an_tree.query(grid_pts[valid_grid], k=1)
    conf_flat[valid_grid] = np.where(dist_v <= res, 0, 1).astype(np.uint8)
    conf_2d = conf_flat.reshape(height, width)

    # write GeoTIFFs
    tif_kwargs = dict(
        driver="GTiff", count=1,
        height=height, width=width,
        crs=profile["crs"], transform=transform,
    )

    for out_path, arr, label in [
        (OUT_BG,   bg_smooth,   "Background"),
        (OUT_RES,  residual_2d, "Residual"),
    ]:
        out = np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32)
        with rasterio.open(out_path, "w", dtype="float32",
                           nodata=-9999.0, **tif_kwargs) as dst:
            dst.write(out, 1)
        v = arr[np.isfinite(arr)]
        print(f"Saved: {out_path.name}")
        print(f"  {label}: {v.min():.1f} ~ {v.max():.1f} nT  std={v.std():.1f} nT")

    with rasterio.open(OUT_CONF, "w", dtype="uint8",
                       nodata=255, **tif_kwargs) as dst:
        dst.write(conf_2d, 1)

    measured     = (conf_2d[conf_2d != 255] == 0).sum()
    interpolated = (conf_2d[conf_2d != 255] == 1).sum()
    total_c      = measured + interpolated
    print(f"Saved: {OUT_CONF.name}")
    if total_c > 0:
        print(f"Confidence: measured={measured} ({100*measured/total_c:.1f}%), "
              f"interpolated={interpolated} ({100*interpolated/total_c:.1f}%)")


if __name__ == "__main__":
    main()