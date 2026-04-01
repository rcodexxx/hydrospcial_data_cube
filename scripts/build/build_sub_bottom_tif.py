# scripts/build/build_rl.py
"""
Build SBP-derived primary layers from seafloor bottom echo.

Outputs:
  sbp_rl.tif           - Reflection Loss (dB), used by build_sediment.py
  sbp_impedance.tif    - Acoustic impedance (Pa·s/m), for report
  sbp_pulse_width.tif  - Bottom echo pulse width at half max (m), for report
  sbp_confidence.tif   - 0=measured, 1=RF predicted, 255=nodata
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
from src.sbp.calculation import RL_MIN, RL_MAX, compute_rl, rl_to_impedance
from src.config import SBP_CC, EPSG, SOUND_SPEED

ROOT     = Path(__file__).parent.parent.parent
SBP_PATH = ROOT / "data/sbp"
MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
VRM_TIF  = ROOT / "outputs/tif/mbes_vrm.tif"
BS_TIF   = ROOT / "outputs/tif/sss_backscatter_lf.tif"
OUT_RL   = ROOT / "outputs/tif/sbp_rl.tif"
OUT_Z    = ROOT / "outputs/tif/sbp_impedance.tif"
OUT_PW   = ROOT / "outputs/tif/sbp_pulse_width.tif"
OUT_CONF = ROOT / "outputs/tif/sbp_confidence.tif"

SDEP   = 20480e-9 * SOUND_SPEED / 2
PW_MIN = 0.01
PW_MAX = 2.0


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


def pulse_width_half_max(amps, idx_b):
    peak = float(amps[idx_b])
    if peak <= 0:
        return None
    half = peak * 0.5

    left = idx_b
    for j in range(idx_b - 1, max(idx_b - 200, 0), -1):
        if amps[j] < half:
            left = j
            break

    right = idx_b
    for j in range(idx_b + 1, min(idx_b + 200, len(amps))):
        if amps[j] < half:
            right = j
            break

    width = right - left
    if width < 2:
        return None
    return width * SDEP


def train_and_predict_rf(all_x, all_y, target, label, clip_range,
                         grid_pts, valid_grid, height, width):
    bathy = sample_raster(MBES_TIF, all_x, all_y)
    vrm   = sample_raster(VRM_TIF, all_x, all_y)
    bs    = sample_raster(BS_TIF, all_x, all_y)

    mask = (np.isfinite(bathy) & np.isfinite(vrm)
            & np.isfinite(bs) & np.isfinite(target))
    X_train = np.column_stack([bathy[mask], vrm[mask], bs[mask]])
    y_train = target[mask]
    print(f"  Training points: {mask.sum()}")

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10,
        min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    print(f"  Feature importance:")
    for name, imp in zip(["Bathymetry", "VRM", "SSS_BS_LF"],
                         rf.feature_importances_):
        print(f"    {name:12s}: {imp:.4f}")

    n_cv = min(5000, len(X_train))
    idx_cv = np.random.choice(len(X_train), n_cv, replace=False)
    cv = cross_val_score(rf, X_train[idx_cv], y_train[idx_cv],
                         cv=5, scoring="r2", n_jobs=-1)
    print(f"  R²: {cv.mean():.3f} ± {cv.std():.3f}")

    bathy_g = sample_raster(MBES_TIF, grid_pts[:, 0], grid_pts[:, 1])
    vrm_g   = sample_raster(VRM_TIF, grid_pts[:, 0], grid_pts[:, 1])
    bs_g    = sample_raster(BS_TIF, grid_pts[:, 0], grid_pts[:, 1])

    valid_f = (valid_grid & np.isfinite(bathy_g)
               & np.isfinite(vrm_g) & np.isfinite(bs_g))
    X_pred = np.column_stack([bathy_g[valid_f], vrm_g[valid_f], bs_g[valid_f]])

    pred = np.full(len(X_pred), np.nan)
    chunk = 50000
    for i in range(0, len(X_pred), chunk):
        e = min(i + chunk, len(X_pred))
        pred[i:e] = rf.predict(X_pred[i:e])

    pred = np.clip(pred, *clip_range)
    out = np.full(len(grid_pts), np.nan)
    out[valid_f] = pred
    return out.reshape(height, width).astype(np.float32)


def main():
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)
    cc = SBP_CC
    print(f"Using calibrated CC = {cc:.6e} ({20 * np.log10(cc):.2f} dB)")

    # 1. Extract RL and PW per ping
    all_x, all_y, all_rl, all_pw = [], [], [], []

    for jsf in tqdm(sorted(SBP_PATH.glob("*.jsf")), desc="Extracting RL + PW"):
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
            idx_b = find_b(amps[i])
            pw = pulse_width_half_max(amps[i], idx_b)

            if rl is None:
                rl = np.nan
            if pw is None or pw < PW_MIN or pw > PW_MAX:
                pw = np.nan

            all_x.append(x[i])
            all_y.append(y[i])
            all_rl.append(rl)
            all_pw.append(pw)

    all_x  = np.array(all_x, dtype=np.float64)
    all_y  = np.array(all_y, dtype=np.float64)
    all_rl = np.array(all_rl, dtype=np.float64)
    all_pw = np.array(all_pw, dtype=np.float64)
    ping_pts = np.column_stack([all_x, all_y])

    rl_valid = np.isfinite(all_rl)
    pw_valid = np.isfinite(all_pw)
    print(f"\nTotal pings: {len(all_x)}")
    print(f"Valid RL:    {rl_valid.sum()} ({100 * rl_valid.mean():.1f}%)")
    print(f"Valid PW:    {pw_valid.sum()} ({100 * pw_valid.mean():.1f}%)")
    print(f"RL median:   {np.nanmedian(all_rl):.2f} dB")
    print(f"PW median:   {np.nanmedian(all_pw):.4f} m")

    # 2. Build output grid
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

    # 3. RF predict RL
    print("\n--- RL ---")
    rl_2d = train_and_predict_rf(
        all_x, all_y, all_rl, "RL", (RL_MIN, RL_MAX),
        grid_pts, valid_grid, height, width)

    # 4. Compute impedance from RL
    print("\nComputing impedance from RL...")
    z_2d = np.full_like(rl_2d, np.nan)
    rl_finite = np.isfinite(rl_2d)
    z_2d[rl_finite] = rl_to_impedance(rl_2d[rl_finite])

    # 5. RF predict PW
    print("\n--- Pulse Width ---")
    pw_2d = train_and_predict_rf(
        all_x, all_y, all_pw, "PW", (PW_MIN, PW_MAX),
        grid_pts, valid_grid, height, width)

    # 6. Confidence mask
    print("\nBuilding confidence mask...")
    conf_flat = np.full(len(grid_pts), 255, dtype=np.uint8)
    ping_tree = KDTree(ping_pts[rl_valid])
    valid_idx = np.where(valid_grid)[0]
    dist, _ = ping_tree.query(grid_pts[valid_idx], k=1)
    conf_flat[valid_idx] = np.where(dist <= res, 0, 1).astype(np.uint8)
    conf_2d = conf_flat.reshape(height, width)

    # 7. Write GeoTIFFs
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=-9999.0)

    outputs = [
        (OUT_RL, rl_2d, "RL (dB)"),
        (OUT_Z,  z_2d,  "Impedance (Pa·s/m)"),
        (OUT_PW, pw_2d, "Pulse Width (m)"),
    ]
    for out_path, arr, label in outputs:
        out = np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32)
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(out, 1)
        v = arr[np.isfinite(arr)]
        print(f"Saved: {out_path}")
        print(f"  {label} range:  {v.min():.4f} ~ {v.max():.4f}")
        print(f"  {label} median: {np.median(v):.4f}")

    out_profile.update(dtype="uint8", nodata=255)
    with rasterio.open(OUT_CONF, "w", **out_profile) as dst:
        dst.write(conf_2d, 1)
    print(f"Saved: {OUT_CONF}")

    measured = (conf_2d[conf_2d != 255] == 0).sum()
    predicted = (conf_2d[conf_2d != 255] == 1).sum()
    total = measured + predicted
    if total > 0:
        print(f"Confidence: measured={measured} ({100 * measured / total:.1f}%), "
              f"predicted={predicted} ({100 * predicted / total:.1f}%)")


if __name__ == "__main__":
    main()