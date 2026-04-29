"""
Build magnetometer layers: anomaly, background, and residual.

Workflow:
  1. Read .mag files; filter by quality + IGRF tolerance
  2. Apply IGRF correction per survey date
  3. Along-track median filter
       - WIN_BG  → background field (local anomalies removed)
       - WIN_ANOM → anomaly field (electronic noise removed)
  4. IDW interpolation to MBES grid within MAX_EXTRAP_M of tracks
  5. Gaussian low-pass on background
  6. Residual = anomaly - smoothed background

Outputs:
  mag_anomaly.tif    : anomaly field (F - IGRF, anomaly-window medfilt)
  mag_background.tif : long-wavelength magnetic background (nT)
  mag_residual.tif   : anomaly - background (UCH target detection)
  mag_confidence.tif : 0=measured, 1=interpolated, 255=nodata
"""
from datetime import datetime

import numpy as np
import ppigrf
import rasterio
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from scipy.spatial import KDTree
from tqdm import tqdm

from src.config import get_config, ROOT
from src.mag.read_mag import read_mag
from src.mag.config import (
    QUALITY_MIN, F_TOLERANCE, BG_OUTLIER_SIGMA,
    WIN_BG, WIN_ANOM,
    MAX_EXTRAP_M, IDW_POWER, IDW_NEIGHBORS, IDW_CHUNK,
    SMOOTH_M, SMOOTH_WEIGHT_THRESHOLD,
)


def get_igrf(lat, lon, date):
    Be, Bn, Bu = ppigrf.igrf(lon, lat, 0.0, date)
    return float(np.sqrt(np.array(Be) ** 2 +
                         np.array(Bn) ** 2 +
                         np.array(Bu) ** 2).item())


def idw(query_pts, known_pts, known_vals, k, power):
    tree = KDTree(known_pts)
    dists, idxs = tree.query(query_pts, k=k)
    dists = np.maximum(dists, 1e-6)
    w = 1.0 / dists ** power
    w /= w.sum(axis=1, keepdims=True)
    return (w * known_vals[idxs]).sum(axis=1)


def medfilt_safe(arr, window):
    """Median filter with odd window size, capped at array length."""
    n = min(window, len(arr))
    if n % 2 == 0:
        n -= 1
    return medfilt(arr, kernel_size=n)


def load_and_filter(survey_dirs, lat_c, lon_c):
    bg_x, bg_y, bg_v = [], [], []
    anom_x, anom_y, anom_v = [], [], []
    n_total_raw = n_total_kept = 0

    for entry in survey_dirs:
        mag_dir = ROOT / entry["path"]
        date = datetime.strptime(entry["date"], "%Y-%m-%d")
        f_igrf = get_igrf(lat_c, lon_c, date)
        f_lo = f_igrf * (1 - F_TOLERANCE)
        f_hi = f_igrf * (1 + F_TOLERANCE)

        print(f"\n{mag_dir.name}: IGRF={f_igrf:.1f} nT  "
              f"valid F range=[{f_lo:.0f}, {f_hi:.0f}]")

        for mag_file in tqdm(sorted(mag_dir.glob("*.mag")),
                             desc=f"  {mag_dir.name}"):
            records = read_mag(mag_file, apply_layback=True)
            if len(records) < WIN_BG:
                continue

            xs = np.array([r["x"] for r in records])
            ys = np.array([r["y"] for r in records])
            f_arr = np.array([r["F_nT"] for r in records])
            q = np.array([r["quality"] for r in records])
            n_total_raw += len(records)

            ok = (q >= QUALITY_MIN) & (f_arr >= f_lo) & (f_arr <= f_hi)
            if ok.sum() < WIN_BG:
                continue

            xs, ys, f_arr = xs[ok], ys[ok], f_arr[ok]
            n_total_kept += len(xs)

            anom_raw = f_arr - f_igrf
            anom_bg_filt = medfilt_safe(anom_raw, WIN_BG)
            anom_an_filt = medfilt_safe(anom_raw, WIN_ANOM)

            bg_x.extend(xs); bg_y.extend(ys); bg_v.extend(anom_bg_filt)
            anom_x.extend(xs); anom_y.extend(ys); anom_v.extend(anom_an_filt)

    print(f"\nTotal records: {n_total_raw} raw → {n_total_kept} kept "
          f"({100*n_total_kept/max(n_total_raw,1):.1f}%)")

    return (np.array(bg_x), np.array(bg_y), np.array(bg_v),
            np.array(anom_x), np.array(anom_y), np.array(anom_v))


def main():
    cfg = get_config()
    mag_cfg = cfg["mag"]
    lat_c = float(mag_cfg["igrf_center"]["lat"])
    lon_c = float(mag_cfg["igrf_center"]["lon"])

    out_bg = ROOT / mag_cfg["outputs"]["background_tif"]
    out_an = ROOT / mag_cfg["outputs"]["anomaly_tif"]
    out_re = ROOT / mag_cfg["outputs"]["residual_tif"]
    out_cf = ROOT / mag_cfg["outputs"]["confidence_tif"]
    mbes_tif = ROOT / cfg["mbes"]["bathymetry_tif"]

    # ── 1. Load and filter ────────────────────────────────
    print("1. Loading and filtering MAG records...")
    bg_x, bg_y, bg_v, anom_x, anom_y, anom_v = load_and_filter(
        mag_cfg["survey_dirs"], lat_c, lon_c
    )

    # Background outlier removal
    bg_std = bg_v.std()
    bg_keep = np.abs(bg_v) < BG_OUTLIER_SIGMA * bg_std
    bg_x, bg_y, bg_v = bg_x[bg_keep], bg_y[bg_keep], bg_v[bg_keep]
    bg_pts = np.column_stack([bg_x, bg_y])
    anom_pts = np.column_stack([anom_x, anom_y])

    print(f"\nBackground pts : {len(bg_x)} "
          f"(removed {(~bg_keep).sum()} outliers)  "
          f"std={bg_v.std():.1f} nT")
    print(f"Anomaly pts    : {len(anom_x)}  std={anom_v.std():.1f} nT")

    # ── 2. Load MBES grid ─────────────────────────────────
    print("\n2. Loading MBES grid geometry...")
    with rasterio.open(mbes_tif) as src:
        profile = src.profile.copy()
        transform = src.transform
        height, width = src.height, src.width
        mbes_data = src.read(1)
        mbes_nodata = src.nodata

    res = transform.a
    xs_grid = transform.c + (np.arange(width) + 0.5) * res
    ys_grid = transform.f + (np.arange(height) + 0.5) * (-res)
    gx, gy = np.meshgrid(xs_grid, ys_grid)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

    valid_grid = (
        (mbes_data != mbes_nodata).ravel()
        if mbes_nodata is not None
        else np.isfinite(mbes_data).ravel()
    )

    # Distance mask: only interpolate within MAX_EXTRAP_M of tracks
    print(f"\n3. Distance mask (MAX_EXTRAP_M = {MAX_EXTRAP_M:.0f} m)...")
    bg_tree = KDTree(bg_pts)
    dist, _ = bg_tree.query(grid_pts[valid_grid], k=1)
    within = dist <= MAX_EXTRAP_M
    vg_idx = np.where(valid_grid)[0]
    valid_grid[vg_idx[~within]] = False

    n_mbes = (mbes_data != mbes_nodata).sum()
    print(f"  Valid grid pts: {valid_grid.sum()} "
          f"({100*valid_grid.sum()/n_mbes:.1f}% of MBES coverage)")

    valid_indices = np.where(valid_grid)[0]

    # ── 4. IDW interpolation: background + anomaly ───────
    print("\n4. IDW interpolation...")
    bg_grid = np.full(len(grid_pts), np.nan)
    for i in tqdm(range(0, len(valid_indices), IDW_CHUNK), desc="  background"):
        idx = valid_indices[i:i + IDW_CHUNK]
        bg_grid[idx] = idw(grid_pts[idx], bg_pts, bg_v,
                           k=IDW_NEIGHBORS, power=IDW_POWER)
    bg_2d = bg_grid.reshape(height, width).astype(np.float32)

    anom_grid = np.full(len(grid_pts), np.nan)
    for i in tqdm(range(0, len(valid_indices), IDW_CHUNK), desc="  anomaly"):
        idx = valid_indices[i:i + IDW_CHUNK]
        anom_grid[idx] = idw(grid_pts[idx], anom_pts, anom_v,
                             k=IDW_NEIGHBORS, power=IDW_POWER)
    anom_2d = anom_grid.reshape(height, width).astype(np.float32)

    # ── 5. Gaussian low-pass on background ───────────────
    sigma_px = SMOOTH_M / res
    print(f"\n5. Gaussian low-pass on background "
          f"(σ = {sigma_px:.0f} px = {SMOOTH_M:.0f} m)...")
    bg_valid = np.isfinite(bg_2d)
    bg_filled = np.where(bg_valid, bg_2d, 0.0)
    bg_smooth = gaussian_filter(bg_filled, sigma=sigma_px)
    weight = gaussian_filter(bg_valid.astype(float), sigma=sigma_px)
    bg_smooth = np.where(weight > SMOOTH_WEIGHT_THRESHOLD,
                         bg_smooth / weight, np.nan)
    bg_smooth = np.where(bg_valid, bg_smooth, np.nan).astype(np.float32)

    # ── 6. Residual = anomaly - background ───────────────
    residual_2d = np.where(
        np.isfinite(anom_2d) & np.isfinite(bg_smooth),
        anom_2d - bg_smooth, np.nan,
    ).astype(np.float32)

    # ── 7. Confidence layer ──────────────────────────────
    print("\n6. Confidence mask...")
    conf_flat = np.full(len(grid_pts), 255, dtype=np.uint8)
    an_tree = KDTree(anom_pts)
    dist_v, _ = an_tree.query(grid_pts[valid_grid], k=1)
    conf_flat[valid_grid] = np.where(dist_v <= res, 0, 1).astype(np.uint8)
    conf_2d = conf_flat.reshape(height, width)

    # ── 8. Write GeoTIFFs ─────────────────────────────────
    print("\n7. Saving GeoTIFFs...")
    tif_kwargs = dict(
        driver="GTiff", count=1,
        height=height, width=width,
        crs=profile["crs"], transform=transform,
    )

    for path, arr, label in [
        (out_an, anom_2d,    "Anomaly"),
        (out_bg, bg_smooth,  "Background"),
        (out_re, residual_2d, "Residual"),
    ]:
        out = np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32)
        with rasterio.open(path, "w", dtype="float32",
                           nodata=-9999.0, **tif_kwargs) as dst:
            dst.write(out, 1)
        v = arr[np.isfinite(arr)]
        print(f"  {path.name:<22} {label:<11} "
              f"{v.min():+.1f} ~ {v.max():+.1f} nT  std={v.std():.1f} nT")

    with rasterio.open(out_cf, "w", dtype="uint8",
                       nodata=255, **tif_kwargs) as dst:
        dst.write(conf_2d, 1)

    measured = (conf_2d[conf_2d != 255] == 0).sum()
    interpolated = (conf_2d[conf_2d != 255] == 1).sum()
    total_c = measured + interpolated
    print(f"  {out_cf.name:<22} measured={measured} "
          f"({100*measured/max(total_c,1):.1f}%), "
          f"interpolated={interpolated} ({100*interpolated/max(total_c,1):.1f}%)")


if __name__ == "__main__":
    main()