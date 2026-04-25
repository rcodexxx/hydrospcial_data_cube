"""
Build sediment classification layer from SBP RL measurements.

Workflow:
  1. Extract RL per ping from SBP JSF (vectorized)
  2. Spike + along-track smoothing
  3. IDW spatial interpolation to MBES grid,
     limited to IDW_MAX_GAP_M around tracklines
  4. Analytical threshold lookup → sediment class

Outputs:
  sbp_rl.tif              - Reflection Loss (dB)
  sbp_sediment_class.tif  - Sediment class index
  sbp_confidence.tif      - 0=measured (within 1 res cell),
                            1=interpolated (within IDW_MAX_GAP_M),
                            255=outside coverage

Rationale:
  SBP RL is the only direct acoustic measurement of sub-bottom reflectivity.
  SSS backscatter is a relative value (per-line normalized, no absolute dB
  calibration) and its correlation with SBP-derived RL cannot be validated
  without ground truth sampling. We therefore use pure spatial interpolation
  on SBP measurements, accepting reduced spatial coverage in exchange for
  methodological defensibility. See src/sbp/calculation.py for RL threshold
  derivation.
"""
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree
from tqdm import tqdm

from src.config import get_config, ROOT
from src.sbp.calculation import (
    classify_sediment, compute_rl_batch, get_sediment_labels,
)
from src.sbp.config import (
    RL_MIN, RL_MAX,
    RL_SPIKE_WINDOW, RL_SPIKE_THRESHOLD, RL_SMOOTH_WINDOW,
    IDW_MAX_GAP_M, IDW_K_NEIGHBORS, IDW_EPS,
)
from src.sbp.read_sbp_jsf import read_sbp_jsf


def main():
    cfg = get_config()
    epsg = cfg["grid"]["epsg"]
    sbp_cc = float(cfg["sbp"]["calibration_constant"])

    mbes_tif = ROOT / cfg["mbes"]["bathymetry_tif"]
    out_rl = ROOT / cfg["sbp"]["outputs"]["rl_tif"]
    out_sed = ROOT / cfg["sbp"]["outputs"]["sediment_tif"]
    out_conf = ROOT / cfg["sbp"]["outputs"]["confidence_tif"]
    sbp_dirs = [ROOT / d["path"] for d in cfg["sbp"]["survey_dirs"]]

    sediment_labels = get_sediment_labels()
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    print(f"Using CC = {sbp_cc:.6e} ({20 * np.log10(sbp_cc):.2f} dB)")
    print(f"IDW max gap: {IDW_MAX_GAP_M} m, k={IDW_K_NEIGHBORS}")
    print(f"Sediment classes: {sediment_labels}")

    # ── 1. Extract RL per ping ─────────────────────────────
    print("\nExtracting RL from SBP pings...")
    all_x, all_y, all_rl = [], [], []

    jsf_files = []
    for d in sbp_dirs:
        jsf_files.extend(sorted(d.glob("*.jsf")))

    for jsf in tqdm(jsf_files, desc="Reading SBP"):
        data = read_sbp_jsf(jsf)
        if "SBP" not in data:
            continue
        d = data["SBP"]
        valid = ~np.isnan(d["lon"])
        if not valid.any():
            continue

        x, y = transformer.transform(d["lon"][valid], d["lat"][valid])
        rls = compute_rl_batch(d["amps"][valid], sbp_cc)

        # spike rejection
        if len(rls) >= RL_SPIKE_WINDOW:
            smooth_ref = median_filter(rls, size=RL_SPIKE_WINDOW)
            spike_mask = np.abs(rls - smooth_ref) > RL_SPIKE_THRESHOLD
            rls[spike_mask] = np.nan

        # along-track smoothing (preserves NaN positions)
        if len(rls) >= RL_SMOOTH_WINDOW:
            finite_mask = np.isfinite(rls)
            if finite_mask.any():
                temp_filled = np.where(finite_mask, rls, np.nanmedian(rls))
                smoothed = median_filter(temp_filled, size=RL_SMOOTH_WINDOW)
                smoothed[~finite_mask] = np.nan
                rls = smoothed

        ok = np.isfinite(rls)
        if not ok.any():
            continue

        all_x.append(x[ok])
        all_y.append(y[ok])
        all_rl.append(rls[ok])

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_rl = np.concatenate(all_rl)
    ping_pts = np.column_stack([all_x, all_y])

    print(f"Valid RL pings : {len(all_rl)}")
    print(f"RL range       : {all_rl.min():.2f} ~ {all_rl.max():.2f} dB")
    print(f"RL median      : {np.median(all_rl):.2f} dB")

    # ── 2. Load MBES grid geometry ──────────────────────────
    print("\nLoading MBES grid...")
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

    # ── 3. IDW interpolation ───────────────────────────────
    max_dist = IDW_MAX_GAP_M
    print(f"\nIDW interpolation (max_dist={max_dist} m, k={IDW_K_NEIGHBORS})...")
    tree = cKDTree(ping_pts)
    target_pts = grid_pts[valid_grid]

    distances, indices = tree.query(
        target_pts, k=IDW_K_NEIGHBORS, distance_upper_bound=max_dist
    )

    interp_rl = np.full(len(target_pts), np.nan, dtype=np.float32)
    nearest_dist = np.full(len(target_pts), np.inf, dtype=np.float32)

    for i in tqdm(range(len(target_pts)), desc="IDW"):
        valid_k = np.isfinite(distances[i]) & (distances[i] <= max_dist)
        if not valid_k.any():
            continue
        d_k = distances[i][valid_k]
        v_k = all_rl[indices[i][valid_k]]
        w = 1.0 / (d_k ** 2 + IDW_EPS)
        interp_rl[i] = np.sum(w * v_k) / np.sum(w)
        nearest_dist[i] = d_k.min()

    interp_rl = np.clip(interp_rl, RL_MIN, RL_MAX)

    # ── 4. Assemble full grid + smooth + classify ──────────
    from scipy.ndimage import gaussian_filter

    # Place IDW result into full grid (NaN outside valid)
    rl_flat = np.full(len(grid_pts), np.nan, dtype=np.float32)
    rl_flat[valid_grid] = interp_rl
    rl_2d_raw = rl_flat.reshape(height, width)

    # Gaussian smooth on RL grid (handles NaN via weighted blur)
    mask = np.isfinite(rl_2d_raw)
    rl_filled = np.where(mask, rl_2d_raw, 0)
    weights = mask.astype(np.float32)

    SMOOTH_SIGMA_PX = 20  # ~4 m at 0.5 m grid
    rl_smoothed = gaussian_filter(rl_filled, sigma=SMOOTH_SIGMA_PX)
    weights_smoothed = gaussian_filter(weights, sigma=SMOOTH_SIGMA_PX)
    rl_2d = np.where(
        weights_smoothed > 0.1,
        rl_smoothed / np.maximum(weights_smoothed, 1e-10),
        np.nan,
    )
    rl_flat = rl_2d.ravel()

    # Classify on smoothed RL
    sed_flat = np.full(len(grid_pts), -1, dtype=np.int8)
    valid_rl = np.isfinite(rl_flat)
    sed_flat[valid_rl] = np.array(
        [classify_sediment(v) for v in rl_flat[valid_rl]], dtype=np.int8
    )
    sed_2d = sed_flat.reshape(height, width)

    # Confidence mask
    conf_flat = np.full(len(grid_pts), 255, dtype=np.uint8)
    nearest_flat = np.full(len(grid_pts), np.inf, dtype=np.float32)
    nearest_flat[valid_grid] = nearest_dist
    conf_flat[valid_rl] = np.where(
        nearest_flat[valid_rl] <= res, 0, 1
    ).astype(np.uint8)
    conf_2d = conf_flat.reshape(height, width)

    # ── 5. Write GeoTIFFs ──────────────────────────────────
    print("\nWriting outputs...")
    tif_kwargs = dict(
        driver="GTiff", count=1,
        height=height, width=width,
        crs=profile["crs"], transform=transform,
    )

    out_rl_data = np.where(np.isfinite(rl_2d), rl_2d, -9999.0).astype(np.float32)
    with rasterio.open(out_rl, "w", dtype="float32", nodata=-9999.0, **tif_kwargs) as dst:
        dst.write(out_rl_data, 1)
    with rasterio.open(out_sed, "w", dtype="int8", nodata=-1, **tif_kwargs) as dst:
        dst.write(sed_2d, 1)
    with rasterio.open(out_conf, "w", dtype="uint8", nodata=255, **tif_kwargs) as dst:
        dst.write(conf_2d, 1)

    # ── Summary ────────────────────────────────────────────
    rl_valid = rl_2d[np.isfinite(rl_2d)]
    print(f"\nSaved: {out_rl.name}, {out_sed.name}, {out_conf.name}")
    print(f"RL range  : {rl_valid.min():.2f} ~ {rl_valid.max():.2f} dB")
    print(f"RL median : {np.median(rl_valid):.2f} dB")

    print("\nSediment class distribution:")
    total = (sed_2d >= 0).sum()
    for i, label in enumerate(sediment_labels):
        count = (sed_2d == i).sum()
        if count > 0:
            print(f"  {i:2d} {label:30s}: {count:7d} px "
                  f"({100 * count / max(total, 1):.1f}%)")

    measured = (conf_2d == 0).sum()
    predicted = (conf_2d == 1).sum()
    total_valid = valid_grid.sum()
    print(f"\nMeasured       : {measured} ({100 * measured / total_valid:.1f}% of valid grid)")
    print(f"IDW interpolated: {predicted} ({100 * predicted / total_valid:.1f}% of valid grid)")
    print(f"Outside coverage: {total_valid - measured - predicted} "
          f"({100 * (total_valid - measured - predicted) / total_valid:.1f}% of valid grid)")


if __name__ == "__main__":
    main()