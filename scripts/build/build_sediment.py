# scripts/build/build_sediment.py
"""
Build sediment classification layer from SBP + SSS + MBES.

Workflow:
  1. Extract RL per ping from SBP JSF (vectorized)
  2. Sample features at ping locations:
     - Bathymetry, VRM (continuous)
     - HF/LF cluster labels (one-hot encoded)
  3. RF regression to predict RL within max_dist of SBP tracks
  4. Hamilton table lookup → sediment class
  5. Confidence mask: 0=measured, 1=interpolated, 255=outside coverage

Outputs:
  sbp_rl.tif              - Reflection Loss (dB), nodata outside SBP coverage
  sbp_sediment_class.tif  - Hamilton sediment classification, nodata outside coverage
  sbp_confidence.tif      - 0=measured, 1=interpolated, 255=nodata
"""
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.spatial import KDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy.ndimage import median_filter
from tqdm import tqdm

from src.data_loader.read_sbp_jsf import read_sbp_jsf
from src.sbp.calculation import (
    SEDIMENT_LABELS, RL_MIN, RL_MAX,
    classify_sediment, compute_rl_batch,
)
from src.config import SBP_CC, EPSG, ROOT

SBP_PATH = ROOT / "data/sbp"
MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
VRM_TIF  = ROOT / "outputs/tif/mbes_vrm.tif"
HF_TIF   = ROOT / "outputs/tif/sss_clusters_hf.tif"
LF_TIF   = ROOT / "outputs/tif/sss_clusters_lf.tif"
OUT_RL   = ROOT / "outputs/tif/sbp_rl.tif"
OUT_SED  = ROOT / "outputs/tif/sbp_sediment_class.tif"
OUT_CONF = ROOT / "outputs/tif/sbp_confidence.tif"

MAX_DIST       = 100.0  # metres: only interpolate within this distance of SBP tracks
N_HF_CLUSTERS = 5
N_LF_CLUSTERS = 5


def sample_raster(tif_path, xs, ys):
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        nd, t = src.nodata, src.transform
        h, w  = data.shape
    cols = ((xs - t.c) / t.a).astype(int)
    rows = ((ys - t.f) / t.e).astype(int)
    v    = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    vals = np.full(len(xs), np.nan, dtype=np.float32)
    vals[v] = data[rows[v], cols[v]]
    if nd is not None:
        vals[vals == nd] = np.nan
    return vals


def cluster_onehot(cluster_vals, n_clusters):
    n = len(cluster_vals)
    oh = np.zeros((n, n_clusters), dtype=np.float32)
    for c in range(n_clusters):
        oh[:, c] = (cluster_vals == c).astype(np.float32)
    return oh


def build_features(xs, ys):
    """Sample all features, return (X, feature_names)."""
    depth = sample_raster(MBES_TIF, xs, ys)
    vrm   = sample_raster(VRM_TIF,  xs, ys)
    hf_c  = sample_raster(HF_TIF,   xs, ys)
    lf_c  = sample_raster(LF_TIF,   xs, ys)

    hf_oh = cluster_onehot(hf_c, N_HF_CLUSTERS)
    lf_oh = cluster_onehot(lf_c, N_LF_CLUSTERS)

    cols  = [depth, vrm] + [hf_oh[:, c] for c in range(N_HF_CLUSTERS)] \
                         + [lf_oh[:, c] for c in range(N_LF_CLUSTERS)]
    names = ["Bathymetry", "VRM"] \
          + [f"HF_C{c}" for c in range(N_HF_CLUSTERS)] \
          + [f"LF_C{c}" for c in range(N_LF_CLUSTERS)]

    return np.column_stack(cols), names


def main():
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)
    print(f"Using calibrated CC = {SBP_CC:.6e} ({20*np.log10(SBP_CC):.2f} dB)")
    print(f"Interpolation radius: {MAX_DIST} m")

    # ── 1. Extract RL per ping (vectorized) ──────────────────
    print("\nExtracting RL from SBP pings...")
    all_x, all_y, all_rl = [], [], []

    for jsf in tqdm(sorted(SBP_PATH.glob("*.jsf")), desc="Reading SBP"):
        data = read_sbp_jsf(jsf)
        if "SBP" not in data:
            continue
        d = data["SBP"]
        valid = ~np.isnan(d["lon"])
        if not valid.any():
            continue

        x, y = transformer.transform(d["lon"][valid], d["lat"][valid])
        rls  = compute_rl_batch(d["amps"][valid], SBP_CC)

        # spike過濾
        if len(rls) >= 21:
            smooth_ref = median_filter(rls, size=21)
            spike_mask = np.abs(rls - smooth_ref) > 5.0
            rls[spike_mask] = np.nan

        # 沿軌平滑
        if len(rls) >= 11:
            finite_mask = np.isfinite(rls)
            if finite_mask.any():
                temp_filled = np.where(finite_mask, rls, np.nanmedian(rls))
                smoothed = median_filter(temp_filled, size=11)
                smoothed[~finite_mask] = np.nan
                rls = smoothed

        ok = np.isfinite(rls)
        if not ok.any():
            continue

        # append在迴圈裡，concatenate在迴圈外
        all_x.append(x[ok])
        all_y.append(y[ok])
        all_rl.append(rls[ok])

    # 迴圈結束後才concatenate
    all_x  = np.concatenate(all_x)
    all_y  = np.concatenate(all_y)
    all_rl = np.concatenate(all_rl)
    ping_pts = np.column_stack([all_x, all_y])

    print(f"Valid RL pings : {len(all_rl)}")
    print(f"RL range       : {all_rl.min():.2f} ~ {all_rl.max():.2f} dB")
    print(f"RL median      : {np.median(all_rl):.2f} dB")

    # ── 2. Sample features at ping locations ─────────────────
    print("\nSampling features at ping locations...")
    X_all, feature_names = build_features(all_x, all_y)

    mask = np.isfinite(all_rl)
    for i, name in enumerate(feature_names):
        if name in ("Bathymetry", "VRM"):
            mask &= np.isfinite(X_all[:, i])

    X_train = X_all[mask]
    y_train = all_rl[mask]
    print(f"Training points: {mask.sum()}")

    # ── 3. Train RF ───────────────────────────────────────────
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12,
        min_samples_leaf=5, n_jobs=-1, random_state=42,
    )
    rf.fit(X_train, y_train)

    print("\nFeature importance:")
    for name, imp in zip(feature_names, rf.feature_importances_):
        if imp > 0.001:
            print(f"  {name:12s}: {imp:.4f}")

    n_cv   = min(5000, len(X_train))
    idx_cv = np.random.choice(len(X_train), n_cv, replace=False)
    cv = cross_val_score(rf, X_train[idx_cv], y_train[idx_cv],
                         cv=5, scoring="r2", n_jobs=-1)
    print(f"R²: {cv.mean():.3f} ± {cv.std():.3f}")

    # ── 4. Build output grid ──────────────────────────────────
    with rasterio.open(MBES_TIF) as src:
        profile     = src.profile.copy()
        transform   = src.transform
        height, width = src.height, src.width
        mbes_data   = src.read(1)
        mbes_nodata = src.nodata

    res = transform.a
    xs_grid = transform.c + (np.arange(width)  + 0.5) * res
    ys_grid = transform.f + (np.arange(height) + 0.5) * (-res)
    gx, gy  = np.meshgrid(xs_grid, ys_grid)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

    valid_grid = (
        (mbes_data != mbes_nodata).ravel()
        if mbes_nodata is not None
        else np.isfinite(mbes_data).ravel()
    )

    # ── 5. Distance filter: only predict within MAX_DIST ─────
    print(f"\nComputing distance to SBP tracks (max {MAX_DIST} m)...")
    ping_tree  = KDTree(ping_pts)
    dist_all   = np.full(len(grid_pts), np.inf)
    query_idx  = np.where(valid_grid)[0]
    dist_valid, _ = ping_tree.query(grid_pts[query_idx], k=1)
    dist_all[query_idx] = dist_valid

    within_coverage = valid_grid & (dist_all <= MAX_DIST)

    covered = within_coverage.sum()
    total   = valid_grid.sum()
    print(f"Grid cells within {MAX_DIST} m: {covered} / {total} "
          f"({100*covered/total:.1f}%)")

    # ── 6. Sample features on grid (within coverage only) ────
    print("\nSampling features on coverage grid...")
    X_grid, _ = build_features(
        grid_pts[within_coverage, 0],
        grid_pts[within_coverage, 1],
    )

    # valid_f: within coverage AND continuous features finite
    cont_ok = np.isfinite(X_grid[:, 0]) & np.isfinite(X_grid[:, 1])
    valid_f_local = cont_ok  # index into within_coverage subset
    valid_f_global = within_coverage.copy()
    valid_f_global[within_coverage] &= cont_ok

    X_pred = X_grid[valid_f_local]

    # ── 7. Predict RL in chunks ───────────────────────────────
    print("Predicting RL...")
    chunk   = 50000
    rl_pred = np.full(len(X_pred), np.nan)
    for i in tqdm(range(0, len(X_pred), chunk), desc="RF predict"):
        e = min(i + chunk, len(X_pred))
        rl_pred[i:e] = rf.predict(X_pred[i:e])

    rl_pred = np.clip(rl_pred, RL_MIN, RL_MAX)

    rl_flat  = np.full(len(grid_pts), np.nan)
    rl_flat[valid_f_global] = rl_pred
    rl_2d = rl_flat.reshape(height, width).astype(np.float32)

    # ── 8. Hamilton lookup → sediment class ──────────────────
    print("Classifying sediment...")
    sed_flat = np.full(len(grid_pts), -1, dtype=np.int8)
    sed_flat[valid_f_global] = np.array(
        [classify_sediment(v) for v in rl_pred], dtype=np.int8
    )
    sed_2d = sed_flat.reshape(height, width)

    # ── 9. Confidence mask ────────────────────────────────────
    # 0 = measured (within res of a ping)
    # 1 = interpolated (within MAX_DIST)
    # 255 = outside SBP coverage or no MBES data
    conf_flat = np.full(len(grid_pts), 255, dtype=np.uint8)
    conf_flat[valid_f_global] = np.where(
        dist_all[valid_f_global] <= res, 0, 1
    ).astype(np.uint8)
    conf_2d = conf_flat.reshape(height, width)

    # ── 10. Write GeoTIFFs ────────────────────────────────────
    tif_kwargs = dict(
        driver="GTiff", count=1,
        height=height, width=width,
        crs=profile["crs"], transform=transform,
    )

    out_rl = np.where(np.isfinite(rl_2d), rl_2d, -9999.0).astype(np.float32)
    with rasterio.open(OUT_RL,  "w", dtype="float32", nodata=-9999.0, **tif_kwargs) as dst:
        dst.write(out_rl, 1)
    with rasterio.open(OUT_SED, "w", dtype="int8",    nodata=-1,      **tif_kwargs) as dst:
        dst.write(sed_2d, 1)
    with rasterio.open(OUT_CONF,"w", dtype="uint8",   nodata=255,     **tif_kwargs) as dst:
        dst.write(conf_2d, 1)

    # ── Summary ───────────────────────────────────────────────
    rl_valid = rl_2d[np.isfinite(rl_2d)]
    print(f"\nSaved: {OUT_RL.name}, {OUT_SED.name}, {OUT_CONF.name}")
    print(f"RL range  : {rl_valid.min():.2f} ~ {rl_valid.max():.2f} dB")
    print(f"RL median : {np.median(rl_valid):.2f} dB")

    print("\nSediment class distribution:")
    total = (sed_2d >= 0).sum()
    for i, label in enumerate(SEDIMENT_LABELS):
        count = (sed_2d == i).sum()
        if count > 0:
            print(f"  {i:2d} {label:25s}: {count:7d} px "
                  f"({100*count/max(total,1):.1f}%)")

    measured  = (conf_2d[conf_2d != 255] == 0).sum()
    predicted = (conf_2d[conf_2d != 255] == 1).sum()
    total_c   = measured + predicted
    if total_c > 0:
        print(f"\nConfidence: measured={measured} ({100*measured/total_c:.1f}%), "
              f"predicted={predicted} ({100*predicted/total_c:.1f}%)")


if __name__ == "__main__":
    main()