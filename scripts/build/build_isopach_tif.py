# scripts/build/build_isopach.py
"""
Build isopach (sediment thickness) layer from SBP data.

Workflow:
  1. Pick bottom echo (B) and sub-bottom reflector (BTB) per ping
  2. Compute thickness from two-way travel time difference
  3. Use RF regression to interpolate along-track thickness to full grid
  4. Generate SNR-based confidence mask

Note: thickness is computed using water sound speed (no V_sed correction).
The error from using water speed vs. sediment speed is <5% for shallow
unconsolidated sediments, which is smaller than the BTB picking uncertainty.

R² is expected to be low (~0) in environments with dredging or artificial
disturbance. The RF output in such cases converges to the spatial mean,
producing a smooth but low-variance surface.
"""
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy.spatial import KDTree
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from src.data_loader.read_sbp_jsf import read_sbp_jsf
from src.config import SOUND_SPEED, EPSG

ROOT      = Path(__file__).parent.parent.parent
SBP_PATH  = ROOT / "data/sbp"
MBES_TIF  = ROOT / "outputs/tif/mbes_bathymetry.tif"
VRM_TIF   = ROOT / "outputs/tif/mbes_vrm.tif"
BS_TIF    = ROOT / "outputs/tif/sss_backscatter_lf.tif"
OUT_THICK = ROOT / "outputs/tif/sbp_isopach.tif"
OUT_CONF  = ROOT / "outputs/tif/sbp_isopach_confidence.tif"

NS         = 20480e-9
SDEP       = NS * SOUND_SPEED / 2
THICK_CLIP = (0.2, 2.0)
SNR_MIN    = 1.0


def sample_raster(tif_path, xs, ys):
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        nd, t = src.nodata, src.transform
        h, w = data.shape
    cols = ((xs - t.c) / t.a).astype(int)
    rows = ((ys - t.f) / t.e).astype(int)
    v = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    vals = np.full(len(xs), np.nan)
    vals[v] = data[rows[v], cols[v]]
    if nd is not None:
        vals[vals == nd] = np.nan
    return vals


def find_b(amps, blanking=50):
    return int(np.argmax(amps[blanking:])) + blanking


def detrend_btb(amps, idx_b):
    """
    Detect sub-bottom reflector (BTB) using detrended residual analysis.
    Returns (snr, thickness_m) or (None, None) if not detected.
    """
    s = idx_b + int(0.3 / SDEP)
    e = min(len(amps), idx_b + int(2.5 / SDEP))
    if e - s < 10:
        return None, None

    window = amps[s:e].astype(np.float64)
    x = np.arange(len(window), dtype=np.float64)
    db_win = 20 * np.log10(np.maximum(window, 1.0))

    slope, intercept, _, _, _ = linregress(x, db_win)
    residual = db_win - (slope * x + intercept)

    local_max = [(i, residual[i])
                 for i in range(1, len(residual) - 1)
                 if residual[i] > residual[i - 1]
                 and residual[i] > residual[i + 1]]
    if not local_max:
        return None, None

    best_i, best_val = max(local_max, key=lambda p: p[1])
    res_std = residual.std()
    snr = best_val / res_std if res_std > 0 else 0
    thick = (s + best_i - idx_b) * SDEP / 2.0
    return snr, thick


def main():
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)

    # 1. Compute thickness per ping
    all_x, all_y, all_thick, all_snr = [], [], [], []
    n_total, n_no_btb, n_low_snr, n_clipped = 0, 0, 0, 0

    for jsf in tqdm(sorted(SBP_PATH.glob("*.jsf")), desc="Computing thickness"):
        data = read_sbp_jsf(jsf)
        if "SBP" not in data:
            continue
        d = data["SBP"]
        valid = ~np.isnan(d["lon"])
        if valid.sum() == 0:
            continue
        x, y = transformer.transform(d["lon"][valid], d["lat"][valid])
        amps = d["amps"][valid].astype(np.float32)

        for i in range(len(amps)):
            n_total += 1
            idx_b = find_b(amps[i])

            snr, thick = detrend_btb(amps[i], idx_b)
            if snr is None:
                n_no_btb += 1
                continue
            if snr < SNR_MIN:
                n_low_snr += 1
                continue
            if thick < THICK_CLIP[0] or thick > THICK_CLIP[1]:
                n_clipped += 1
                continue

            all_x.append(x[i])
            all_y.append(y[i])
            all_thick.append(thick)
            all_snr.append(snr)

    all_x     = np.array(all_x, dtype=np.float64)
    all_y     = np.array(all_y, dtype=np.float64)
    all_thick = np.array(all_thick, dtype=np.float64)
    all_snr   = np.array(all_snr, dtype=np.float64)
    ping_pts  = np.column_stack([all_x, all_y])

    print(f"\nTotal pings      : {n_total}")
    print(f"No BTB detected  : {n_no_btb} ({100 * n_no_btb / n_total:.1f}%)")
    print(f"Low SNR (<{SNR_MIN})   : {n_low_snr} ({100 * n_low_snr / n_total:.1f}%)")
    print(f"Out of range     : {n_clipped} ({100 * n_clipped / n_total:.1f}%)")
    print(f"Valid thickness  : {len(all_thick)} ({100 * len(all_thick) / n_total:.1f}%)")
    print(f"Thickness range  : {all_thick.min():.3f} ~ {all_thick.max():.3f} m")
    print(f"Thickness median : {np.median(all_thick):.3f} m")
    print(f"SNR median       : {np.median(all_snr):.2f}")

    # 2. Sample features at ping locations
    print("\nSampling features at ping locations...")
    bathy = sample_raster(MBES_TIF, all_x, all_y)
    vrm   = sample_raster(VRM_TIF, all_x, all_y)
    bs    = sample_raster(BS_TIF, all_x, all_y)

    mask = (np.isfinite(bathy) & np.isfinite(vrm)
            & np.isfinite(bs) & np.isfinite(all_thick))
    feature_names = ["Bathymetry", "VRM", "SSS_BS_LF"]
    X_train = np.column_stack([bathy[mask], vrm[mask], bs[mask]])
    y_train = all_thick[mask]
    print(f"Training points: {mask.sum()}")

    # 3. Train Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    print("\nFeature importance:")
    for name, imp in zip(feature_names, rf.feature_importances_):
        print(f"  {name:12s}: {imp:.4f}")

    print("\nCross-validation (5-fold)...")
    n_cv = min(5000, len(X_train))
    idx_cv = np.random.choice(len(X_train), n_cv, replace=False)
    cv_scores = cross_val_score(
        rf, X_train[idx_cv], y_train[idx_cv], cv=5, scoring="r2", n_jobs=-1
    )
    r2_mean = cv_scores.mean()
    print(f"Mean R²: {r2_mean:.3f} ± {cv_scores.std():.3f}")
    if r2_mean < 0.1:
        print("WARNING: Low R², terrain features poorly predict thickness.")
        print("         RF output will converge to spatial mean.")

    # 4. Read MBES grid for output geometry
    with rasterio.open(MBES_TIF) as src:
        profile   = src.profile.copy()
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

    # 5. Sample features on full grid
    print("\nSampling features on full grid...")
    bathy_g = sample_raster(MBES_TIF, grid_pts[:, 0], grid_pts[:, 1])
    vrm_g   = sample_raster(VRM_TIF, grid_pts[:, 0], grid_pts[:, 1])
    bs_g    = sample_raster(BS_TIF, grid_pts[:, 0], grid_pts[:, 1])

    valid_features = (valid_grid & np.isfinite(bathy_g)
                      & np.isfinite(vrm_g) & np.isfinite(bs_g))

    X_pred = np.column_stack([
        bathy_g[valid_features],
        vrm_g[valid_features],
        bs_g[valid_features],
    ])

    # 6. Predict thickness
    print("Predicting thickness on full grid...")
    chunk_size = 50000
    n_chunks = int(np.ceil(len(X_pred) / chunk_size))
    thick_pred = np.full(len(X_pred), np.nan, dtype=np.float32)

    for i in tqdm(range(n_chunks), desc="RF predict"):
        s = i * chunk_size
        e = min(s + chunk_size, len(X_pred))
        thick_pred[s:e] = rf.predict(X_pred[s:e])

    thick_pred = np.clip(thick_pred, *THICK_CLIP)

    thick_flat = np.full(len(grid_pts), np.nan, dtype=np.float32)
    thick_flat[valid_features] = thick_pred
    thick_2d = thick_flat.reshape(height, width)

    # 7. SNR-based confidence mask
    print("Building confidence mask...")
    conf_flat = np.zeros(len(grid_pts), dtype=np.uint8)

    ping_tree = KDTree(ping_pts)
    dist, idx = ping_tree.query(grid_pts[valid_features], k=1)

    for j, vf_idx in enumerate(np.where(valid_features)[0]):
        d = dist[j]
        if d <= res:
            snr_val = all_snr[idx[j]]
            conf_flat[vf_idx] = int(np.clip(128 + (snr_val / 3.0) * 127, 128, 255))
        else:
            max_dist = 50.0
            conf_flat[vf_idx] = int(np.clip(127 * (1 - d / max_dist), 1, 127))

    conf_2d = conf_flat.reshape(height, width)

    # 8. Write GeoTIFFs
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=-9999.0)

    out = np.where(np.isfinite(thick_2d), thick_2d, -9999.0).astype(np.float32)
    with rasterio.open(OUT_THICK, "w", **out_profile) as dst:
        dst.write(out, 1)
    print(f"Saved: {OUT_THICK}")

    out_profile.update(dtype="uint8", nodata=0)
    with rasterio.open(OUT_CONF, "w", **out_profile) as dst:
        dst.write(conf_2d, 1)
    print(f"Saved: {OUT_CONF}")

    # Summary
    thick_v = thick_2d[np.isfinite(thick_2d)]
    if len(thick_v):
        print(f"\nThickness range  : {thick_v.min():.3f} ~ {thick_v.max():.3f} m")
        print(f"Thickness median : {np.median(thick_v):.3f} m")

    conf_valid = conf_2d[conf_2d > 0]
    if len(conf_valid):
        measured = (conf_valid >= 128).sum()
        predicted = (conf_valid < 128).sum()
        print(f"Confidence: measured={measured} ({100 * measured / len(conf_valid):.1f}%), "
              f"predicted={predicted} ({100 * predicted / len(conf_valid):.1f}%)")


if __name__ == "__main__":
    main()