# scripts/build/build_sediment.py
"""
Build sediment classification layer from SBP data.

Workflow:
  1. Extract RL per ping from JSF (CC calibration)
  2. RF regression to interpolate RL to full grid
     Features: [bathymetry, VRM, BS_LF, BS_HF]
  3. Hamilton table lookup → sediment class
  4. Confidence mask: measured vs RF predicted

Outputs:
  sbp_rl.tif              - Reflection Loss (dB)
  sbp_sediment_class.tif  - Hamilton sediment classification (int8)
  sbp_confidence.tif      - 0=measured, 1=RF predicted, 255=nodata
"""
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy.spatial import KDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from src.data_loader.read_sbp_jsf import read_sbp_jsf
from src.sbp.calculation import (SEDIMENT_LABELS, RL_MIN, RL_MAX,
                                 classify_sediment, compute_rl)
from src.config import SBP_CC, EPSG, ROOT

SBP_PATH = ROOT / "data/sbp"
MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
VRM_TIF  = ROOT / "outputs/tif/mbes_vrm.tif"
BS_LF    = ROOT / "outputs/tif/sss_backscatter_lf.tif"
BS_HF    = ROOT / "outputs/tif/sss_backscatter_hf.tif"
OUT_RL   = ROOT / "outputs/tif/sbp_rl.tif"
OUT_SED  = ROOT / "outputs/tif/sbp_sediment_class.tif"
OUT_CONF = ROOT / "outputs/tif/sbp_confidence.tif"

FEATURE_TIFS  = [MBES_TIF, VRM_TIF, BS_LF, BS_HF]
FEATURE_NAMES = ["Bathymetry", "VRM", "BS_LF", "BS_HF"]


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


def main():
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)
    cc = SBP_CC
    print(f"Using calibrated CC = {cc:.6e} ({20 * np.log10(cc):.2f} dB)")

    # 1. Extract RL per ping
    all_x, all_y, all_rl = [], [], []

    for jsf in tqdm(sorted(SBP_PATH.glob("*.jsf")), desc="Extracting RL"):
        data = read_sbp_jsf(jsf)
        if "SBP" not in data:
            continue
        d = data["SBP"]
        valid = ~np.isnan(d["lon"])
        if valid.sum() == 0:
            continue
        x, y = transformer.transform(d["lon"][valid], d["lat"][valid])
        amps = d["amps"][valid]

        for i in range(len(amps)):
            rl = compute_rl(amps[i], cc)
            if rl is None:
                continue
            all_x.append(x[i])
            all_y.append(y[i])
            all_rl.append(rl)

    all_x  = np.array(all_x, dtype=np.float64)
    all_y  = np.array(all_y, dtype=np.float64)
    all_rl = np.array(all_rl, dtype=np.float64)
    ping_pts = np.column_stack([all_x, all_y])

    print(f"\nValid RL pings : {len(all_rl)}")
    print(f"RL range       : {all_rl.min():.2f} ~ {all_rl.max():.2f} dB")
    print(f"RL median      : {np.median(all_rl):.2f} dB")

    # 2. Sample features at ping locations
    print("\nSampling features at ping locations...")
    features = [sample_raster(tif, all_x, all_y) for tif in FEATURE_TIFS]
    mask = np.isfinite(all_rl)
    for f in features:
        mask &= np.isfinite(f)

    X_train = np.column_stack([f[mask] for f in features])
    y_train = all_rl[mask]
    print(f"Training points: {mask.sum()}")

    # 3. Train RF
    print("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10,
        min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    print("\nFeature importance:")
    for name, imp in zip(FEATURE_NAMES, rf.feature_importances_):
        print(f"  {name:12s}: {imp:.4f}")

    n_cv = min(5000, len(X_train))
    idx_cv = np.random.choice(len(X_train), n_cv, replace=False)
    cv = cross_val_score(rf, X_train[idx_cv], y_train[idx_cv],
                         cv=5, scoring="r2", n_jobs=-1)
    print(f"R²: {cv.mean():.3f} ± {cv.std():.3f}")

    # 4. Build output grid
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

    # 5. Sample features on grid
    print("\nSampling features on full grid...")
    grid_features = [sample_raster(tif, grid_pts[:, 0], grid_pts[:, 1])
                     for tif in FEATURE_TIFS]
    valid_f = valid_grid.copy()
    for f in grid_features:
        valid_f &= np.isfinite(f)

    X_pred = np.column_stack([f[valid_f] for f in grid_features])

    # 6. Predict RL
    print("Predicting RL...")
    chunk = 50000
    rl_pred = np.full(len(X_pred), np.nan)
    for i in tqdm(range(0, len(X_pred), chunk), desc="RF predict"):
        e = min(i + chunk, len(X_pred))
        rl_pred[i:e] = rf.predict(X_pred[i:e])

    rl_pred = np.clip(rl_pred, RL_MIN, RL_MAX)
    rl_flat = np.full(len(grid_pts), np.nan)
    rl_flat[valid_f] = rl_pred
    rl_2d = rl_flat.reshape(height, width).astype(np.float32)

    # 7. Hamilton lookup → sediment class
    print("Classifying sediment...")
    sed_flat = np.full(len(grid_pts), -1, dtype=np.int8)
    sed_flat[valid_f] = np.array([classify_sediment(v) for v in rl_pred],
                                  dtype=np.int8)
    sed_2d = sed_flat.reshape(height, width)

    # 8. Confidence mask
    print("Building confidence mask...")
    conf_flat = np.full(len(grid_pts), 255, dtype=np.uint8)
    ping_tree = KDTree(ping_pts)
    dist, _ = ping_tree.query(grid_pts[np.where(valid_f)[0]], k=1)
    conf_flat[valid_f] = np.where(dist <= res, 0, 1).astype(np.uint8)
    conf_2d = conf_flat.reshape(height, width)

    # 9. Write GeoTIFFs
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=-9999.0)
    out_rl = np.where(np.isfinite(rl_2d), rl_2d, -9999.0).astype(np.float32)
    with rasterio.open(OUT_RL, "w", **out_profile) as dst:
        dst.write(out_rl, 1)
    print(f"Saved: {OUT_RL}")

    out_profile.update(dtype="int8", count=1, nodata=-1)
    with rasterio.open(OUT_SED, "w", **out_profile) as dst:
        dst.write(sed_2d, 1)
    print(f"Saved: {OUT_SED}")

    out_profile.update(dtype="uint8", nodata=255)
    with rasterio.open(OUT_CONF, "w", **out_profile) as dst:
        dst.write(conf_2d, 1)
    print(f"Saved: {OUT_CONF}")

    # Summary
    rl_valid = rl_2d[np.isfinite(rl_2d)]
    print(f"\nRL range  : {rl_valid.min():.2f} ~ {rl_valid.max():.2f} dB")
    print(f"RL median : {np.median(rl_valid):.2f} dB")

    print("\nSediment class distribution:")
    total = (sed_2d >= 0).sum()
    for i, label in enumerate(SEDIMENT_LABELS):
        count = (sed_2d == i).sum()
        if count > 0:
            print(f"  {i:2d} {label:25s}: {count:7d} px ({100 * count / max(total, 1):.1f}%)")

    measured = (conf_2d[conf_2d != 255] == 0).sum()
    predicted = (conf_2d[conf_2d != 255] == 1).sum()
    total_c = measured + predicted
    if total_c > 0:
        print(f"\nConfidence: measured={measured} ({100 * measured / total_c:.1f}%), "
              f"predicted={predicted} ({100 * predicted / total_c:.1f}%)")


if __name__ == "__main__":
    main()