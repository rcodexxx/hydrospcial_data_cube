# scripts/build/build_sbp_tif.py
"""
Build SBP-derived layers: RL and sediment classification.
Uses RF regression with [bathymetry, VRM, SSS backscatter] to
interpolate along-track RL to full grid.

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
from src.config import SBP_CC, EPSG

ROOT     = Path(__file__).parent.parent.parent
SBP_PATH = ROOT / "data/sbp"
MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
VRM_TIF  = ROOT / "outputs/tif/mbes_vrm.tif"
BS_TIF   = ROOT / "outputs/tif/sss_backscatter_lf.tif"
OUT_RL   = ROOT / "outputs/tif/sbp_rl.tif"
OUT_SED  = ROOT / "outputs/tif/sbp_sediment_class.tif"
OUT_CONF = ROOT / "outputs/tif/sbp_confidence.tif"


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

    # 1. Compute RL for all pings
    all_x, all_y, all_rl = [], [], []

    for jsf in tqdm(sorted(SBP_PATH.glob("*.jsf")), desc="Computing RL"):
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

    print(f"Valid RL pings : {len(all_rl)}")
    print(f"RL range       : {all_rl.min():.2f} ~ {all_rl.max():.2f} dB")
    print(f"RL median      : {np.median(all_rl):.2f} dB")

    # 2. Sample features at ping locations
    print("Sampling features at ping locations...")
    bathy = sample_raster(MBES_TIF, all_x, all_y)
    vrm   = sample_raster(VRM_TIF, all_x, all_y)
    bs    = sample_raster(BS_TIF, all_x, all_y)

    mask = (np.isfinite(bathy) & np.isfinite(vrm)
            & np.isfinite(bs) & np.isfinite(all_rl))
    feature_names = ["Bathymetry", "VRM", "SSS_BS_LF"]
    X_train = np.column_stack([bathy[mask], vrm[mask], bs[mask]])
    y_train = all_rl[mask]
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
    print(f"Mean R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # 4. Read MBES grid for output geometry
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

    # 6. Predict RL
    print("Predicting RL on full grid...")
    chunk_size = 50000
    n_chunks = int(np.ceil(len(X_pred) / chunk_size))
    rl_pred = np.full(len(X_pred), np.nan)

    for i in tqdm(range(n_chunks), desc="RF predict"):
        s = i * chunk_size
        e = min(s + chunk_size, len(X_pred))
        rl_pred[s:e] = rf.predict(X_pred[s:e])

    rl_pred = np.clip(rl_pred, RL_MIN, RL_MAX)

    rl_grid = np.full(len(grid_pts), np.nan)
    rl_grid[valid_features] = rl_pred
    rl_2d = rl_grid.reshape(height, width).astype(np.float32)

    # 7. Sediment classification
    print("Classifying sediment...")
    sed_grid = np.full(len(grid_pts), -1, dtype=np.int8)
    finite = np.isfinite(rl_grid)
    sed_grid[finite] = [classify_sediment(v)
                        for v in tqdm(rl_grid[finite], desc="Classify")]
    sed_2d = sed_grid.reshape(height, width)

    # 8. Confidence mask
    conf_grid = np.full(len(grid_pts), 255, dtype=np.uint8)
    dist, _ = KDTree(ping_pts).query(grid_pts[valid_features], k=1)
    conf_grid[valid_features] = np.where(dist <= res, 0, 1)
    conf_2d = conf_grid.reshape(height, width)

    # 9. Write GeoTIFFs
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=-9999.0)

    out_rl = np.where(np.isfinite(rl_2d), rl_2d, -9999.0).astype(np.float32)
    with rasterio.open(OUT_RL, "w", **out_profile) as dst:
        dst.write(out_rl, 1)
    print(f"Saved: {OUT_RL}")

    out_profile.update(dtype="int8", nodata=-1)
    with rasterio.open(OUT_SED, "w", **out_profile) as dst:
        dst.write(sed_2d, 1)
    print(f"Saved: {OUT_SED}")

    out_profile.update(dtype="uint8", nodata=255)
    with rasterio.open(OUT_CONF, "w", **out_profile) as dst:
        dst.write(conf_2d, 1)
    print(f"Saved: {OUT_CONF}")

    # Summary
    print("\nSediment class distribution:")
    total = (sed_2d >= 0).sum()
    for i, label in enumerate(SEDIMENT_LABELS):
        count = (sed_2d == i).sum()
        print(f"  {i} {label:20s}: {count:6d} px ({100 * count / max(total, 1):.1f}%)")

    rl_valid = rl_2d[np.isfinite(rl_2d)]
    if len(rl_valid):
        print(f"\nRL range  : {rl_valid.min():.2f} ~ {rl_valid.max():.2f} dB")
        print(f"RL median : {np.median(rl_valid):.2f} dB")


if __name__ == "__main__":
    main()