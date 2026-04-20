# scripts/build/build_backscatter.py
"""
Build hydrospatial SSS backscatter and acoustic facies mosaics.

Workflow:
  Pass 1a: Georef + collect raw BS → build global ARC
  Pass 1b: Ping-level features → fit K-means → predict + smooth labels
  Pass 2:  Per-cluster ARC correction → per-line median statistics
  Pass 3:  Port/Stbd balancing + cross-line normalization + physical shift
           → weighted accumulation
  Pass 4:  IDW hole filling + DEM masking → output GeoTIFF

Outputs:
  sss_backscatter_{freq}.tif  — calibrated backscatter mosaic (dB)
  sss_clusters_{freq}.tif     — acoustic facies labels
"""
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
from tqdm import tqdm

from src.backscatter.accumulation import WeightedAccumulator
from src.backscatter.correction import (
    apply_arc_correction, build_global_arc, collect_features,
    fit_kmeans, predict_labels,
)
from src.backscatter.georef import georef_line
from src.config import EPSG, ROOT

# ==========================================
# Configuration
# ==========================================
FREQ         = "lf"
CHANNELS     = ["LF_port", "LF_stbd"]
N_CLUSTERS   = 5
PHYSICAL_SHIFT_DB = -25.22 if FREQ == "lf" else -26.99

MBES_TIF    = ROOT / "outputs/tif/mbes_bathymetry.tif"
OUT_BS_TIF  = ROOT / f"outputs/tif/sss_backscatter_{FREQ}.tif"
OUT_LBL_TIF = ROOT / f"outputs/tif/sss_clusters_{FREQ}.tif"

SURVEY_DIRS = {
    ROOT / "data/sss/20251223": 15.0,
    ROOT / "data/sss/20251224": None,
}

GEOREF_KWARGS = dict(
    turn_threshold=3,
    turn_cooldown=0.0,
    roll_threshold=5.0,
    heading_rate_threshold=3.0,
)


# ==========================================
# Main Pipeline
# ==========================================
def main():
    jsf_files = [
        (f, cable)
        for d, cable in SURVEY_DIRS.items()
        for f in sorted(d.glob("*.jsf"))
    ]
    print(f"JSF files: {len(jsf_files)}, channels: {CHANNELS}")

    # preload MBES once to avoid repeated IO
    with rasterio.open(MBES_TIF) as src:
        bounds   = src.bounds
        dem      = src.read(1).astype(np.float32)
        dem_mask = np.isnan(dem)
        if src.nodata is not None:
            dem_mask |= (dem == src.nodata)
        mbes_preloaded = {
            "data":      src.read(1).astype(np.float32),
            "transform": src.transform,
            "tr": Transformer.from_crs(
                "EPSG:4326", f"EPSG:{src.crs.to_epsg()}", always_xy=True
            ),
        }

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)

    # ── Pass 1a: Georef + global ARC ─────────────────────────
    print("\nPass 1a: Georef + building global ARC...")
    raw_bs_all, raw_inc_all = [], []
    all_results, result_sides, global_raw_bs = [], [], []

    for jsf_path, cable in tqdm(jsf_files, desc="Pass 1a"):
        for ch in CHANNELS:
            r = georef_line(jsf_path, MBES_TIF, ch,
                            cable_length=cable,
                            mbes_preloaded=mbes_preloaded,
                            **GEOREF_KWARGS)
            if r is None:
                continue

            bs_db = 10 * np.log10(np.maximum(r["bs"], 1e-12))
            raw_bs_all.append(bs_db)
            raw_inc_all.append(r["inc_angle"])

            valid_bs = r["bs"][r["bs"] > 0]
            if len(valid_bs):
                global_raw_bs.append(valid_bs)

            xs, ys = tr.transform(r["lon"], r["lat"])
            all_results.append((r, xs.astype(np.float32), ys.astype(np.float32)))
            result_sides.append("port" if "port" in ch else "stbd")

    if not all_results:
        print("No valid data found.")
        return

    bin_centers_arc, arc_curve = build_global_arc(
        np.concatenate(raw_bs_all),
        np.concatenate(raw_inc_all),
    )
    print("Global ARC built.")

    merged_raw = np.concatenate(global_raw_bs)
    min_db = np.percentile(10 * np.log10(merged_raw.astype(np.float64)), 1.0)
    print(f"BS_MIN_DB: {min_db:.2f} dB")

    # ── Pass 1b: Ping features + K-means ─────────────────────
    print(f"\nPass 1b: Ping features + K-means (K={N_CLUSTERS})...")
    sample_feats, feat_cache = [], {}

    for i, ((r, xs, ys), side) in enumerate(
            tqdm(zip(all_results, result_sides), desc="Pass 1b", total=len(all_results))):
        feat, valid_pings = collect_features(
            r["bs"], r["inc_angle"], r["ping_idx"], bin_centers_arc, arc_curve
        )
        if feat is not None:
            sample_feats.append(feat)
            feat_cache[i] = (feat, valid_pings)

    if not sample_feats:
        print("No valid features found.")
        return

    km = fit_kmeans(sample_feats, N_CLUSTERS)

    line_labels_map = {
        i: predict_labels(km, feat, vp)
        for i, (feat, vp) in feat_cache.items()
    }

    # ── Pass 2: ARC correction + median collection ────────────
    print("\nPass 2: ARC correction + median collection...")
    port_medians, stbd_medians, corrected_results = [], [], []

    for i, ((r, xs, ys), side) in enumerate(
            tqdm(zip(all_results, result_sides), desc="Pass 2", total=len(all_results))):
        if i not in line_labels_map:
            continue

        valid_pings, ping_labels = line_labels_map[i]
        bs_corr, bs_raw, labels = apply_arc_correction(
            r["bs"], r["inc_angle"], r["ping_idx"],
            valid_pings, ping_labels,
            bin_centers_arc, arc_curve,
            n_clusters=N_CLUSTERS, nadir_cutoff=12,
        )
        bs_corr[bs_corr < min_db] = np.nan

        valid = np.isfinite(bs_corr)
        med = float(np.median(bs_corr[valid])) if valid.any() else np.nan

        (port_medians if side == "port" else stbd_medians).append(med)
        corrected_results.append(
            (bs_corr, bs_raw, labels, xs, ys, r["inc_angle"], side, med)
        )

    # Port/Stbd balancing
    port_grand  = np.nanmedian(port_medians)
    stbd_grand  = np.nanmedian(stbd_medians)
    grand_median = np.nanmedian(port_medians + stbd_medians)
    port_offset  = grand_median - port_grand
    stbd_offset  = grand_median - stbd_grand

    print(f"\nPort/Stbd balancing:")
    print(f"  Port: {port_grand:.1f} dB  (offset: {port_offset:+.1f} dB)")
    print(f"  Stbd: {stbd_grand:.1f} dB  (offset: {stbd_offset:+.1f} dB)")
    print(f"  Grand median: {grand_median:.1f} dB")

    # ── Pass 3: Normalization + accumulation ──────────────────
    print("\nPass 3: Normalization + accumulation...")
    acc = WeightedAccumulator(bounds)

    for bs_corr, bs_raw, labels, xs, ys, inc_angle, side, line_med in tqdm(
            corrected_results, desc="Pass 3"):
        if not np.isfinite(line_med):
            continue

        side_offset  = port_offset if side == "port" else stbd_offset
        adjusted_med = line_med + side_offset
        bs_corr = bs_corr + side_offset - adjusted_med + grand_median + PHYSICAL_SHIFT_DB

        acc.add(xs, ys, bs_corr, bs_raw=bs_raw, inc_angle=inc_angle, labels=labels)

    out_bs, _, out_lbl = acc.result()

    # ── Pass 4: IDW hole filling + DEM masking ────────────────
    print("\nPass 4: IDW hole filling...")
    valid_mask_global = (out_bs != -9999.0) & ~dem_mask
    hole_mask         = (out_bs == -9999.0) & ~dem_mask

    if hole_mask.any() and valid_mask_global.any():
        known_pts  = np.column_stack(np.where(valid_mask_global)[::-1])
        query_pts  = np.column_stack(np.where(hole_mask)[::-1])
        tree       = cKDTree(known_pts)

        dists, indices = tree.query(query_pts, k=8, distance_upper_bound=7, workers=-1)
        valid_k    = np.isfinite(dists)
        safe_idx   = np.minimum(indices, len(known_pts) - 1)

        known_bs   = out_bs[valid_mask_global]
        vals_bs    = known_bs[safe_idx]
        weights    = np.where(valid_k, 1.0 / (dists ** 2 + 1e-6), 0.0)
        sum_w      = weights.sum(axis=1)
        has_nbr    = sum_w > 0

        idw_bs = np.full(len(query_pts), -9999.0, dtype=np.float32)
        idw_bs[has_nbr] = (weights * vals_bs).sum(axis=1)[has_nbr] / sum_w[has_nbr]
        hy, hx = np.where(hole_mask)
        out_bs[hy, hx] = idw_bs

        # nearest-neighbour for labels
        dists_nn, idx_nn = tree.query(query_pts, k=1, distance_upper_bound=3.0, workers=-1)
        valid_nn = np.isfinite(dists_nn)
        if valid_nn.any():
            known_lbl = out_lbl[valid_mask_global]
            out_lbl[hy[valid_nn], hx[valid_nn]] = known_lbl[idx_nn[valid_nn]]

    out_bs[dem_mask]  = -9999.0
    out_lbl[dem_mask] = 255

    # ── Write GeoTIFFs ────────────────────────────────────────
    OUT_BS_TIF.parent.mkdir(parents=True, exist_ok=True)
    tif_kwargs = dict(
        driver="GTiff", height=acc.nrows, width=acc.ncols,
        count=1, crs=f"EPSG:{EPSG}", transform=acc.transform,
    )
    with rasterio.open(OUT_BS_TIF, "w", dtype="float32", nodata=-9999.0, **tif_kwargs) as dst:
        dst.write(out_bs, 1)
    with rasterio.open(OUT_LBL_TIF, "w", dtype="uint8", nodata=255, **tif_kwargs) as dst:
        dst.write(out_lbl, 1)

    valid_bs = out_bs[out_bs != -9999.0]
    print(f"\nSaved: {OUT_BS_TIF.name}, {OUT_LBL_TIF.name}")
    print(f"BS range:  {valid_bs.min():.1f} ~ {valid_bs.max():.1f} dB")
    print(f"BS median: {np.median(valid_bs):.1f} dB")
    print(f"Coverage:  {100 * len(valid_bs) / (acc.nrows * acc.ncols):.1f}%")


if __name__ == "__main__":
    main()