# scripts/build/build_backscatter.py
"""
Build SSS backscatter mosaic and raw imagery.

Workflow:
  Pass 1: Georef + collect angular profiles for k-means
  Pass 2: Angular correction (Zhao 2017, k-means + polyfit)
  Pass 3: Cross-line normalization + grid accumulation

Outputs:
  sss_backscatter_{freq}.tif  - Angular-corrected backscatter (dB)
  sss_imagery_{freq}.tif      - Raw backscatter without correction (dB)

Note: No topographic correction is applied because EdgeTech firmware
already applies TVG (proc_flags=1). Additional topographic correction
would cause over-compensation.
"""
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from scipy.ndimage import uniform_filter1d, binary_erosion
from tqdm import tqdm

from src.backscatter.georef import georef_line
from src.backscatter.correction import (
    collect_features, fit_kmeans, angle_correction)
from src.config import EPSG, RESOLUTION, ROOT

FREQ       = "hf"                          # "lf" or "hf"
CHANNELS   = ["HF_port", "HF_stbd"]       # ["HF_port", "HF_stbd"] for HF

MBES_TIF    = ROOT / "outputs/tif/mbes_bathymetry.tif"
OUT_BS_TIF  = ROOT / f"outputs/tif/sss_backscatter_{FREQ}.tif"
OUT_IMG_TIF = ROOT / f"outputs/tif/sss_imagery_{FREQ}.tif"
OUT_LBL_TIF = ROOT / f"outputs/tif/sss_clusters_{FREQ}.tif"

SURVEY_DIRS = {
    ROOT / "data/sss/20251223": 15.0,
    ROOT / "data/sss/20251224": None,   # side-mounted, no layback
}

N_CLUSTERS = 7


class GridAccumulator:
    def __init__(self, bounds):
        self.min_x = bounds.left
        self.max_y = bounds.top
        self.ncols = int(round((bounds.right - bounds.left) / RESOLUTION))
        self.nrows = int(round((bounds.top - bounds.bottom) / RESOLUTION))

        self.best_score = np.full((self.nrows, self.ncols), np.inf, dtype=np.float64)
        self.db_best = np.full((self.nrows, self.ncols), np.nan, dtype=np.float64)
        self.label_best = np.full((self.nrows, self.ncols), 255, dtype=np.uint8)

    def add(self, xs, ys, bs_db, inc_angle, labels=None):
        col = np.clip(((xs - self.min_x) / RESOLUTION).astype(np.int32), 0, self.ncols - 1)
        row = np.clip(((self.max_y - ys) / RESOLUTION).astype(np.int32), 0, self.nrows - 1)
        valid = np.isfinite(bs_db)
        for j in np.where(valid)[0]:
            r, c = row[j], col[j]

            score = abs(inc_angle[j] - 45.0)

            if score < self.best_score[r, c]:
                self.best_score[r, c] = score
                self.db_best[r, c] = bs_db[j]
                if labels is not None:
                    self.label_best[r, c] = labels[j]

    def result(self):
        out = self.db_best.astype(np.float32)
        out[~np.isfinite(out)] = -9999.0
        return out

    def result_labels(self):
        return self.label_best

    @property
    def transform(self):
        return from_origin(self.min_x, self.max_y, RESOLUTION, RESOLUTION)


def main():
    jsf_files = [(f, cable)
                 for d, cable in SURVEY_DIRS.items()
                 for f in sorted(d.glob("*.jsf"))]
    print(f"JSF files: {len(jsf_files)}, channels: {CHANNELS}")

    with rasterio.open(MBES_TIF) as src:
        bounds = src.bounds
        dem = src.read(1).astype(np.float32)
        dem_mask = np.isnan(dem)
        if src.nodata is not None:
            dem_mask |= (dem == src.nodata)

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)

    # ── Pass 1: georef + collect angular profiles ─────────────
    print("\nPass 1: georef + angular profile collection...")
    sample_feats = []
    sample_bs_list = []
    sample_inc_list = []
    all_results = []
    result_sides = []
    global_raw_bs = []

    for jsf_path, cable in tqdm(jsf_files, desc="Pass 1"):
        for ch in CHANNELS:
            r = georef_line(jsf_path, MBES_TIF, ch,
                            cable_length=cable,
                            turn_threshold=5)
            if r is None:
                continue

            xs, ys = tr.transform(r["lon"], r["lat"])

            feat, bs_s, inc_s = collect_features(r["bs"], r["inc_angle"])
            sample_feats.append(feat)
            sample_bs_list.append(bs_s)
            sample_inc_list.append(inc_s)

            valid_bs = r["bs"][r["bs"] > 0]
            if len(valid_bs) > 0:
                global_raw_bs.append(valid_bs)

            all_results.append((r, xs.astype(np.float32), ys.astype(np.float32)))
            result_sides.append("port" if "port" in ch else "stbd")

    if not sample_feats:
        print("No valid data found.")
        return

    # dynamic noise floor
    print("\nCalculating dynamic noise floor (1st percentile)...")
    if global_raw_bs:
        merged_raw = np.concatenate(global_raw_bs)
        global_db = 10 * np.log10(merged_raw.astype(np.float64))
        min_db = np.percentile(global_db, 1.0)
        print(f"BS_MIN_DB: {min_db:.2f} dB")

    print(f"\nFitting k-means ({N_CLUSTERS} clusters) on "
          f"{sum(len(p) for p in sample_feats)} samples...")
    km = fit_kmeans(sample_feats, N_CLUSTERS)

    print("\nPlotting true Angular Response Curves (ARC)...")
    import matplotlib.pyplot as plt

    all_feats = np.concatenate(sample_feats)
    all_bs = np.concatenate(sample_bs_list)
    all_inc = np.concatenate(sample_inc_list)

    sample_labels = km.predict(all_feats)
    unique, counts = np.unique(sample_labels, return_counts=True)
    print(f"\nSample Distribution :{dict(zip(unique, counts))}")

    plt.figure(figsize=(9, 6))
    bins = np.arange(15, 86, 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for c in range(N_CLUSTERS):
        mask_c = (sample_labels == c)
        curve = np.full(len(bins) - 1, np.nan)

        for j in range(len(bins) - 1):
            in_bin = mask_c & (all_inc >= bins[j]) & (all_inc < bins[j + 1])
            if in_bin.sum() >= 5:
                curve[j] = np.median(all_bs[in_bin])

        valid = np.isfinite(curve)
        if valid.sum() > 3:
            curve[valid] = uniform_filter1d(curve[valid], size=3)

        cmap = plt.get_cmap('tab20')
        plt.plot(bin_centers, curve, label=f'Cluster {c}', color=cmap(c % 20), linewidth=2)

    plt.title("True Angular Response Curves by K-means Clusters", fontsize=14)
    plt.xlabel("Incidence Angle (degrees)", fontsize=12)
    plt.ylabel("Backscatter Strength (dB)", fontsize=12)
    plt.xlim(15, 85)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    out_sig_fig = ROOT / "outputs/figures/true_arc_signatures.png"
    plt.savefig(out_sig_fig, dpi=150, bbox_inches='tight')
    print(f"Saved ARC Signatures to: {out_sig_fig}")
    plt.close()

    # ── Pass 2: angle correction ──────────────────────────────
    print("\nPass 2: angle correction...")
    port_medians = []
    stbd_medians = []
    corrected_results = []

    for (r, xs, ys), side in tqdm(
            zip(all_results, result_sides), desc="Pass 2",
            total=len(all_results)):
        bs_corr, bs_raw, labels = angle_correction(
            r["bs"], r["inc_angle"], km, N_CLUSTERS, nadir_cutoff=12)

        bs_corr[bs_corr < min_db] = np.nan

        valid = np.isfinite(bs_corr)
        med = np.median(bs_corr[valid]) if valid.any() else np.nan

        if side == "port":
            port_medians.append(med)
        else:
            stbd_medians.append(med)

        corrected_results.append(
            (bs_corr, bs_raw, labels, xs, ys, r["inc_angle"], side, med))

    # ── Port/Stbd balancing ───────────────────────────────────
    port_grand = np.nanmedian(port_medians)
    stbd_grand = np.nanmedian(stbd_medians)
    grand_median = np.nanmedian(port_medians + stbd_medians)

    port_offset = grand_median - port_grand
    stbd_offset = grand_median - stbd_grand

    print(f"Port/Stbd balancing:")
    print(f"  Port grand median: {port_grand:.1f} dB (offset: {port_offset:+.1f} dB)")
    print(f"  Stbd grand median: {stbd_grand:.1f} dB (offset: {stbd_offset:+.1f} dB)")
    print(f"  Grand median:      {grand_median:.1f} dB")

    # ── Pass 3: normalization + grid accumulation ─────────────
    print("\nPass 3: grid accumulation with normalization...")
    acc = GridAccumulator(bounds)
    acc_img = GridAccumulator(bounds)

    for bs_corr, bs_raw, labels, xs, ys, inc_angle, side, line_med in tqdm(
            corrected_results, desc="Pass 3"):

        if not np.isfinite(line_med):
            continue

        # step 1: port/stbd balancing
        side_offset = port_offset if side == "port" else stbd_offset
        bs_corr = bs_corr + side_offset

        # step 2: cross-line normalization
        valid = np.isfinite(bs_corr)
        adjusted_med = np.median(bs_corr[valid]) if valid.any() else line_med

        bs_corr = bs_corr - adjusted_med + grand_median

        acc.add(xs, ys, bs_corr, inc_angle=inc_angle, labels=labels)
        acc_img.add(xs, ys, bs_raw, inc_angle=inc_angle)

    # apply MBES mask
    out_bs = acc.result()
    out_bs[dem_mask] = -9999.0

    out_img = acc_img.result()
    out_img[dem_mask] = -9999.0

    out_lbl = acc.result_labels()
    out_lbl[dem_mask] = 255

    valid_mask = (out_bs != -9999.0)
    eroded_mask = binary_erosion(valid_mask, iterations=3)

    out_bs[~eroded_mask] = -9999.0
    out_img[~eroded_mask] = -9999.0
    out_lbl[~eroded_mask] = 255

    # write backscatter
    with rasterio.open(OUT_BS_TIF, "w", driver="GTiff",
                       height=acc.nrows, width=acc.ncols,
                       count=1, dtype="float32",
                       crs=f"EPSG:{EPSG}",
                       transform=acc.transform,
                       nodata=-9999.0) as dst:
        dst.write(out_bs, 1)
    print(f"\nSaved: {OUT_BS_TIF}")

    # write raw imagery
    with rasterio.open(OUT_IMG_TIF, "w", driver="GTiff",
                       height=acc_img.nrows, width=acc_img.ncols,
                       count=1, dtype="float32",
                       crs=f"EPSG:{EPSG}",
                       transform=acc_img.transform,
                       nodata=-9999.0) as dst:
        dst.write(out_img, 1)
    print(f"Saved: {OUT_IMG_TIF}")

    with rasterio.open(OUT_LBL_TIF, "w", driver="GTiff",
                       height=acc.nrows, width=acc.ncols,
                       count=1, dtype="uint8",
                       crs=f"EPSG:{EPSG}",
                       transform=acc.transform,
                       nodata=255) as dst:
        dst.write(out_lbl, 1)
    print(f"Saved: {OUT_LBL_TIF}")

    # summary
    valid_bs = out_bs[out_bs != -9999.0]
    if len(valid_bs):
        print(f"\nBS range : {valid_bs.min():.1f} ~ {valid_bs.max():.1f} dB")
        print(f"BS median: {np.median(valid_bs):.1f} dB")
        print(f"Coverage : {len(valid_bs)} / {acc.nrows * acc.ncols} cells "
              f"({100 * len(valid_bs) / (acc.nrows * acc.ncols):.1f}%)")


if __name__ == "__main__":
    main()