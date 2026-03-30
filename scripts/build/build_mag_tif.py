# scripts/build_mag_tif_v2.py
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

MAG_DIRS = {
    Path("../../data/mag/20251223"): datetime(2025, 12, 23),
    Path("../../data/mag/20260108"): datetime(2026, 1, 8),
}
MBES_TIF = Path("../../outputs/tif/mbes_bathymetry.tif")
OUT_BG = Path("../../outputs/tif/mag_background.tif")
OUT_RESIDUAL = Path("../../outputs/tif/mag_residual.tif")
OUT_CONF = Path("../../outputs/tif/mag_confidence_v2.tif")

LAT_C = 22.137
LON_C = 120.787
MAX_EXTRAP_M = 30.0
IDW_POWER = 2
IDW_NEIGHBORS = 12

# Background: large window to remove all local anomalies
WIN_BG = 101  # ~75m, keeps only geological background
# Anomaly: small window to remove only electronic noise
WIN_ANOM = 21  # ~15m, preserves UCH anomalies
# Gaussian smooth for background field
SMOOTH_M = 200.0


def get_igrf(date):
    Be, Bn, Bu = ppigrf.igrf(LON_C, LAT_C, 0.0, date)
    return np.sqrt(np.array(Be) ** 2 + np.array(Bn) ** 2 + np.array(Bu) ** 2).item()


def idw(query_pts, known_pts, known_vals, k, power):
    tree = KDTree(known_pts)
    dists, idxs = tree.query(query_pts, k=k)
    dists = np.maximum(dists, 1e-6)
    w = 1.0 / dists**power
    w /= w.sum(axis=1, keepdims=True)
    return (w * known_vals[idxs]).sum(axis=1)


def load_and_filter(mag_dirs, win_bg, win_anom):
    """
    Load all MAG records, apply along-track median filter per file.
    Returns two sets of points:
      - bg_pts/bg_anom: large-window filtered (for background field)
      - anom_pts/anom_vals: small-window filtered (for anomaly field)
    """
    bg_x, bg_y, bg_v = [], [], []
    anom_x, anom_y, anom_v = [], [], []

    for mag_dir, date in mag_dirs.items():
        F_igrf = get_igrf(date)
        print(f"\n{mag_dir.name}: IGRF={F_igrf:.3f} nT")

        for mag_file in tqdm(sorted(mag_dir.glob("*.mag")), desc=f"  {mag_dir.name}"):
            records = read_mag(mag_file, apply_layback=True)
            if len(records) < 5:
                continue

            xs = np.array([r["x"] for r in records])
            ys = np.array([r["y"] for r in records])
            F_raw = np.array([r["F_nT"] for r in records])
            qual = np.array([r["quality"] for r in records])

            # Quality filter
            valid = qual >= 50
            if valid.sum() < 5:
                continue

            xs, ys, F_raw = xs[valid], ys[valid], F_raw[valid]

            # Compute anomaly
            anom_raw = F_raw - F_igrf

            # Remove physically impossible values
            ok = np.abs(anom_raw) < F_igrf * 0.05
            if ok.sum() < 5:
                continue
            xs, ys, anom_raw = xs[ok], ys[ok], anom_raw[ok]

            # Along-track median filter for background (large window)
            win_bg_actual = min(win_bg, len(anom_raw))
            if win_bg_actual % 2 == 0:
                win_bg_actual -= 1
            anom_bg = medfilt(anom_raw, kernel_size=win_bg_actual)

            # Along-track median filter for anomaly (small window)
            win_an_actual = min(win_anom, len(anom_raw))
            if win_an_actual % 2 == 0:
                win_an_actual -= 1
            anom_an = medfilt(anom_raw, kernel_size=win_an_actual)

            bg_x.extend(xs)
            bg_y.extend(ys)
            bg_v.extend(anom_bg)
            anom_x.extend(xs)
            anom_y.extend(ys)
            anom_v.extend(anom_an)

    return (
        np.array(bg_x),
        np.array(bg_y),
        np.array(bg_v),
        np.array(anom_x),
        np.array(anom_y),
        np.array(anom_v),
    )


def main():
    print("Loading and filtering MAG data...")
    bg_x, bg_y, bg_v, anom_x, anom_y, anom_v = load_and_filter(
        MAG_DIRS, WIN_BG, WIN_ANOM
    )

    # Extra cleanup: remove residual strong values after along-track filtering
    bg_std = bg_v.std()
    bg_clean = np.abs(bg_v) < 2.0 * bg_std
    bg_x = bg_x[bg_clean]
    bg_y = bg_y[bg_clean]
    bg_v = bg_v[bg_clean]
    bg_pts = np.column_stack([bg_x, bg_y])
    anom_pts = np.column_stack([anom_x, anom_y])

    print(
        f"Background pts : {len(bg_x)} "
        f"(removed {(~bg_clean).sum()}), std={bg_v.std():.1f} nT"
    )
    print(f"Anomaly pts    : {len(anom_x)}, std={anom_v.std():.1f} nT")

    # Read MBES grid
    with rasterio.open(MBES_TIF) as src:
        profile = src.profile.copy()
        transform = src.transform
        height, width = src.height, src.width
        mbes_data = src.read(1)
        mbes_nodata = src.nodata

    res = transform.a
    xs_grid = transform.c + (np.arange(width) + 0.5) * res
    ys_grid = transform.f + (np.arange(height) + 0.5) * (-res)
    grid_x, grid_y = np.meshgrid(xs_grid, ys_grid)
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    if mbes_nodata is not None:
        valid_grid = (mbes_data != mbes_nodata).ravel()
    else:
        valid_grid = np.isfinite(mbes_data).ravel()

    # Distance mask using background points
    print("\nComputing distance mask...")
    bg_tree = KDTree(bg_pts)
    dist, _ = bg_tree.query(grid_pts[valid_grid], k=1)
    within = dist <= MAX_EXTRAP_M
    vg_idx = np.where(valid_grid)[0]
    valid_grid[vg_idx[~within]] = False
    print(f"Valid grid pts : {valid_grid.sum()}")

    valid_indices = np.where(valid_grid)[0]
    chunk_size = 50000

    # Interpolate background field
    print("\nInterpolating background field...")
    bg_grid = np.full(len(grid_pts), np.nan)
    for i in tqdm(range(0, len(valid_indices), chunk_size), desc="IDW background"):
        idx = valid_indices[i : i + chunk_size]
        bg_grid[idx] = idw(
            grid_pts[idx], bg_pts, bg_v, k=IDW_NEIGHBORS, power=IDW_POWER
        )

    bg_2d = bg_grid.reshape(height, width).astype(np.float32)

    # Gaussian low-pass filter on background
    sigma_px = SMOOTH_M / res
    print(f"Gaussian filter (sigma={sigma_px:.0f} px = {SMOOTH_M:.0f}m)...")
    bg_valid = np.isfinite(bg_2d)
    bg_filled = np.where(bg_valid, bg_2d, 0.0)
    bg_smooth = gaussian_filter(bg_filled, sigma=sigma_px)
    weight = gaussian_filter(bg_valid.astype(float), sigma=sigma_px)
    bg_smooth = np.where(weight > 0.1, bg_smooth / weight, np.nan)
    bg_smooth = np.where(bg_valid, bg_smooth, np.nan).astype(np.float32)

    # Interpolate anomaly field
    print("\nInterpolating anomaly field...")
    an_tree = KDTree(anom_pts)
    anom_grid = np.full(len(grid_pts), np.nan)
    for i in tqdm(range(0, len(valid_indices), chunk_size), desc="IDW anomaly"):
        idx = valid_indices[i : i + chunk_size]
        anom_grid[idx] = idw(
            grid_pts[idx], anom_pts, anom_v, k=IDW_NEIGHBORS, power=IDW_POWER
        )

    anom_2d = anom_grid.reshape(height, width).astype(np.float32)

    # Residual = anomaly - background
    residual_2d = np.where(
        np.isfinite(anom_2d) & np.isfinite(bg_smooth), anom_2d - bg_smooth, np.nan
    ).astype(np.float32)

    # Confidence mask
    conf_grid = np.full(len(grid_pts), 255, dtype=np.uint8)
    dist_v, _ = an_tree.query(grid_pts[valid_grid], k=1)
    conf_grid[valid_grid] = np.where(dist_v <= res, 0, 1)
    conf_2d = conf_grid.reshape(height, width)

    # Write GeoTIFFs
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=-9999.0)

    for out_path, arr in [(OUT_BG, bg_smooth), (OUT_RESIDUAL, residual_2d)]:
        out = arr.copy()
        out[~np.isfinite(arr)] = -9999.0
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(out, 1)

    out_profile.update(dtype="uint8", nodata=255)
    with rasterio.open(OUT_CONF, "w", **out_profile) as dst:
        dst.write(conf_2d, 1)

    print(f"\nSaved: {OUT_BG}, {OUT_RESIDUAL}, {OUT_CONF}")

    bg_v2 = bg_smooth[np.isfinite(bg_smooth)]
    res_v2 = residual_2d[np.isfinite(residual_2d)]
    print(
        f"Background : {bg_v2.min():.1f} ~ {bg_v2.max():.1f} nT, "
        f"std={bg_v2.std():.1f}"
    )
    print(
        f"Residual   : {res_v2.min():.1f} ~ {res_v2.max():.1f} nT, "
        f"std={res_v2.std():.1f}"
    )


if __name__ == "__main__":
    main()
