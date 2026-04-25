"""
Build hybrid isopach: measured (strict SBR detection) + modeled (sediment class).

Where strict SBR is detected → measured thickness from picking.
Elsewhere → modeled thickness from sediment classification × reservoir age.

Outputs:
  - isopach.tif: thickness (m)
  - isopach_source.tif: 1 = measured, 0 = modeled
"""
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from tqdm import tqdm

from src.config import get_config, ROOT
from src.sbp.read_sbp_jsf import read_sbp_jsf
from src.sbp.calculation import compute_rl_batch, rl_to_vp
from src.sbp.sbr_picking import find_bottom, pick_sbr_thickness_strict
from src.sbp.config import (
    THICK_MIN_M, THICK_MAX_M,
    IDW_MAX_GAP_M, IDW_K_NEIGHBORS, IDW_EPS,
)

# Modeled thickness per sediment class (m, 30-year cumulative)
# Class index matches build_sbp_sediment.py output
# 0=Coarse sand ... 6=Fluid mud
MODELED_THICKNESS_M = {
    0: 0.05,   # Coarse sand (dredged / exposed bedrock)
    1: 0.15,   # Fine sand
    2: 0.30,   # Silt / Sandy silt
    3: 0.50,   # Sand-silt-clay
    4: 0.80,   # Compacted mud
    5: 1.20,   # Clayey silt (typical reservoir accumulation)
    6: 2.00,   # Fluid mud (deep stagnant areas, max accumulation)
}

GRID_SMOOTH_SIGMA_PX = 8  # Gaussian sigma for measured-region smoothing
MEASURED_BLEND_RADIUS_PX = 20  # blend radius around measured pixels (~10 m)


def main():
    cfg = get_config()
    epsg = cfg["grid"]["epsg"]
    sbp_cc = float(cfg["sbp"]["calibration_constant"])

    sed_tif = ROOT / cfg["sbp"]["outputs"]["sediment_tif"]
    out_thick = ROOT / cfg["sbp"]["outputs"]["isopach_tif"]
    out_source = out_thick.with_name(out_thick.stem + "_source.tif")
    sbp_dirs = [ROOT / d["path"] for d in cfg["sbp"]["survey_dirs"]]
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    # ── 1. Strict SBR picking per ping ───────────────────
    print("1. Strict SBR detection...")
    all_x, all_y, all_thick, all_snr = [], [], [], []
    n_total = n_picked = 0

    jsf_files = []
    for d in sbp_dirs:
        jsf_files.extend(sorted(d.glob("*.jsf")))

    for jsf in tqdm(jsf_files, desc="Processing JSF"):
        data = read_sbp_jsf(jsf)
        if "SBP" not in data:
            continue
        d = data["SBP"]
        valid = ~np.isnan(d["lon"])
        if not valid.any():
            continue

        x, y = transformer.transform(d["lon"][valid], d["lat"][valid])
        amps = d["amps"][valid].astype(np.float32)
        rls = compute_rl_batch(amps, sbp_cc)

        n_total += len(amps)
        for i in range(len(amps)):
            local_v = rl_to_vp(rls[i])
            idx_b = find_bottom(amps[i])
            result = pick_sbr_thickness_strict(amps[i], idx_b, local_v)
            if result is not None:
                thick = result['thickness_m']
                if THICK_MIN_M <= thick <= THICK_MAX_M:
                    all_x.append(x[i])
                    all_y.append(y[i])
                    all_thick.append(thick)
                    all_snr.append(result['snr'])
                    n_picked += 1

    print(f"\nStrict SBR detection summary:")
    print(f"  Total pings              : {n_total}")
    print(f"  Strict SBR detected      : {n_picked} "
        f"({100*n_picked/max(n_total,1):.1f}%)")

    if n_picked > 0:
        thick_arr = np.array(all_thick)
        snr_arr = np.array(all_snr)
        print(f"  Measured thickness range : {thick_arr.min():.2f} ~ {thick_arr.max():.2f} m")
        print(f"  Measured thickness median: {np.median(thick_arr):.2f} m")
        print(f"  SNR range                : {snr_arr.min():.1f} ~ {snr_arr.max():.1f} σ")
        print(f"  SNR median               : {np.median(snr_arr):.1f} σ")


    # ── 2. Load sediment classification ──────────────────
    print("\n2. Loading sediment classification grid...")
    with rasterio.open(sed_tif) as src:
        sed_class = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        height, width = src.height, src.width
        sed_nodata = src.nodata if src.nodata is not None else -1

    res = transform.a
    valid_grid = (sed_class != sed_nodata)

    # ── 3. Build modeled thickness layer ─────────────────
    print("\n3. Building modeled layer (sediment class → thickness)...")
    modeled = np.full(sed_class.shape, np.nan, dtype=np.float32)
    for class_idx, t in MODELED_THICKNESS_M.items():
        modeled[sed_class == class_idx] = t

    print(f"  Modeled coverage: {np.isfinite(modeled).sum()} cells")
    print(f"  Modeled median  : {np.nanmedian(modeled):.2f} m")

    # ── 4. Build measured layer (IDW from strict picks) ──
    measured = np.full(sed_class.shape, np.nan, dtype=np.float32)

    if n_picked >= 30:  # need a few measurements to interpolate
        print("\n4. Interpolating measured points to grid (local IDW)...")
        xs_grid = transform.c + (np.arange(width) + 0.5) * res
        ys_grid = transform.f + (np.arange(height) + 0.5) * (-res)
        gx, gy = np.meshgrid(xs_grid, ys_grid)

        target_pts = np.column_stack([gx.ravel()[valid_grid.ravel()],
                                       gy.ravel()[valid_grid.ravel()]])

        tree = cKDTree(np.column_stack([all_x, all_y]))
        max_dist = IDW_MAX_GAP_M  # tighter than full IDW; only fill near measurements
        distances, indices = tree.query(
            target_pts, k=IDW_K_NEIGHBORS, distance_upper_bound=max_dist
        )

        all_thick_arr = np.array(all_thick, dtype=np.float32)
        mask = np.isfinite(distances) & (distances <= max_dist)
        weights = np.where(mask, 1.0 / (distances ** 2 + IDW_EPS), 0.0)
        safe_idx = np.where(mask, indices, 0)
        values = all_thick_arr[safe_idx]
        w_sum = weights.sum(axis=1)
        v_sum = (weights * values).sum(axis=1)
        interp = np.where(
            w_sum > 0, v_sum / np.maximum(w_sum, 1e-10), np.nan
        ).astype(np.float32)

        measured_flat = np.full(gx.size, np.nan, dtype=np.float32)
        measured_flat[valid_grid.ravel()] = interp
        measured = measured_flat.reshape(height, width)
        print(f"  Measured coverage on grid: "
              f"{np.isfinite(measured).sum()} cells "
              f"({100*np.isfinite(measured).sum()/valid_grid.sum():.1f}%)")
    else:
        print(f"\n4. Skipping measured layer (only {n_picked} strict picks).")

    # ── 5. Combine: measured wins where present ──────────
    print("\n5. Combining measured + modeled into hybrid isopach...")
    hybrid = np.where(np.isfinite(measured), measured, modeled)
    source = np.where(np.isfinite(measured), 1, 0).astype(np.uint8)
    source[~valid_grid] = 255  # nodata

    # ── 6. Smooth measured boundaries ────────────────────
    if np.isfinite(measured).any():
        print("6. Smoothing measured-region boundaries...")
        mask = np.isfinite(hybrid) & valid_grid
        filled = np.where(mask, hybrid, 0)
        weights = mask.astype(np.float32)
        smoothed = gaussian_filter(filled, sigma=GRID_SMOOTH_SIGMA_PX)
        w_sm = gaussian_filter(weights, sigma=GRID_SMOOTH_SIGMA_PX)
        hybrid = np.where(
            (w_sm > 0.1) & valid_grid,
            smoothed / np.maximum(w_sm, 1e-10),
            np.nan,
        )

    # ── 7. Write outputs ─────────────────────────────────
    print("\n7. Saving GeoTIFFs...")
    thick_out = np.where(np.isfinite(hybrid), hybrid, -9999.0).astype(np.float32)
    profile_t = profile.copy()
    profile_t.update(dtype="float32", count=1, nodata=-9999.0)
    with rasterio.open(out_thick, "w", **profile_t) as dst:
        dst.write(thick_out, 1)

    profile_s = profile.copy()
    profile_s.update(dtype="uint8", count=1, nodata=255)
    with rasterio.open(out_source, "w", **profile_s) as dst:
        dst.write(source, 1)

    valid_t = hybrid[np.isfinite(hybrid)]
    n_meas = (source == 1).sum()
    n_mod = (source == 0).sum()
    print(f"\nSaved:")
    print(f"  Thickness: {out_thick.name}")
    print(f"  Source   : {out_source.name}")
    print(f"  Range          : {valid_t.min():.2f} ~ {valid_t.max():.2f} m")
    print(f"  Median         : {np.median(valid_t):.2f} m")
    print(f"  Measured cells : {n_meas} ({100*n_meas/(n_meas+n_mod):.1f}%)")
    print(f"  Modeled cells  : {n_mod} ({100*n_mod/(n_meas+n_mod):.1f}%)")


if __name__ == "__main__":
    main()