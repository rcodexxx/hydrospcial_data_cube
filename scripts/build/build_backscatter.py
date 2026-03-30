# scripts/build/build_backscatter.py
from pathlib import Path
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from tqdm import tqdm

from src.backscatter.georef import georef_line
from src.backscatter.correction import collect_features, fit_kmeans, angle_correction
from src.config import EPSG, RESOLUTION

ROOT     = Path(__file__).parent.parent.parent
MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
OUT_TIF  = ROOT / "outputs/tif/sss_backscatter_hf.tif"

SURVEY_DIRS = {
    ROOT / "data/sss/20251223": 15.0,
    ROOT / "data/sss/20251224": None,
}

CHANNELS   = ["HF_port", "HF_stbd"]
N_CLUSTERS = 7
BS_MIN_DB  = -50.0


class GridAccumulator:
    def __init__(self, bounds):
        self.min_x   = bounds.left
        self.max_y   = bounds.top
        self.ncols   = int(round((bounds.right - bounds.left) / RESOLUTION))
        self.nrows   = int(round((bounds.top   - bounds.bottom) / RESOLUTION))
        self.min_gr  = np.full((self.nrows, self.ncols), np.inf, dtype=np.float64)
        self.db_best = np.full((self.nrows, self.ncols), np.nan, dtype=np.float64)

    def add(self, xs, ys, bs_db, ground_m):
        col = np.clip(((xs - self.min_x) / RESOLUTION).astype(np.int32),
                      0, self.ncols - 1)
        row = np.clip(((self.max_y - ys) / RESOLUTION).astype(np.int32),
                      0, self.nrows - 1)
        valid = np.isfinite(bs_db)
        for j in np.where(valid)[0]:
            r, c = row[j], col[j]
            if ground_m[j] < self.min_gr[r, c]:
                self.min_gr[r, c] = ground_m[j]
                self.db_best[r, c] = bs_db[j]

    def result(self):
        out = self.db_best.astype(np.float32)
        out[~np.isfinite(out)] = -9999.0
        return out

    @property
    def transform(self):
        return from_origin(self.min_x, self.max_y, RESOLUTION, RESOLUTION)


def main():
    jsf_files = [(f, cable)
                 for d, cable in SURVEY_DIRS.items()
                 for f in sorted(d.glob("*.jsf"))]
    print(f"JSF files: {len(jsf_files)}, channels: {CHANNELS}")

    with rasterio.open(MBES_TIF) as src:
        bounds   = src.bounds
        dem      = src.read(1).astype(np.float32)
        dem_mask = np.isnan(dem)
        if src.nodata is not None:
            dem_mask |= (dem == src.nodata)

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)

    # ── Pass 1: georef + collect angular profiles for k-means ─
    print("\nPass 1: georef + angular profile collection...")
    sample_feats = []
    all_results     = []
    result_sides    = []

    for jsf_path, cable in tqdm(jsf_files, desc="Pass 1"):
        for ch in CHANNELS:
            r = georef_line(jsf_path, MBES_TIF, ch, cable_length=cable)
            if r is None:
                continue

            sample_feats.append(collect_features(r["bs"], r["inc_angle"]))
            xs, ys = tr.transform(r["lon"], r["lat"])
            all_results.append((r, xs.astype(np.float32), ys.astype(np.float32)))
            result_sides.append("port" if "port" in ch else "stbd")

    if not sample_feats:
        print("No valid data found.")
        return

    print(f"\nFitting k-means ({N_CLUSTERS} clusters) on "
          f"{sum(len(p) for p in sample_feats)} angular profiles...")
    km = fit_kmeans(sample_feats, N_CLUSTERS)

    # ── Pass 2: angle correction ──────────────────────────────
    print("\nPass 2: angle correction...")
    port_medians = []
    stbd_medians = []
    corrected_results = []

    for (r, xs, ys), side in tqdm(
            zip(all_results, result_sides), desc="Pass 2",
            total=len(all_results)):
        bs_db, _ = angle_correction(r["bs"], r["inc_angle"], km, N_CLUSTERS)

        bs_db[bs_db < BS_MIN_DB] = np.nan

        valid = np.isfinite(bs_db)
        med = np.median(bs_db[valid]) if valid.any() else np.nan

        if side == "port":
            port_medians.append(med)
        else:
            stbd_medians.append(med)

        corrected_results.append((bs_db, xs, ys, r["ground_m"], side, med))

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

    for bs_db, xs, ys, gm, side, line_med in tqdm(
            corrected_results, desc="Pass 3"):

        if not np.isfinite(line_med):
            continue

        # step 1: port/stbd balancing only
        side_offset = port_offset if side == "port" else stbd_offset
        bs_db = bs_db + side_offset

        # step 2: cross-line normalization (recalculate median after balancing)
        valid = np.isfinite(bs_db)
        adjusted_med = np.median(bs_db[valid]) if valid.any() else line_med
        bs_db = bs_db - adjusted_med + grand_median

        acc.add(xs, ys, bs_db, ground_m=gm)

    # apply MBES mask
    out = acc.result()
    out[dem_mask] = -9999.0

    with rasterio.open(OUT_TIF, "w", driver="GTiff",
                       height=acc.nrows, width=acc.ncols,
                       count=1, dtype="float32",
                       crs=f"EPSG:{EPSG}",
                       transform=acc.transform,
                       nodata=-9999.0) as dst:
        dst.write(out, 1)

    print(f"\nSaved: {OUT_TIF}")

    valid = out[out != -9999.0]
    if len(valid):
        print(f"BS range : {valid.min():.1f} ~ {valid.max():.1f} dB")
        print(f"BS median: {np.median(valid):.1f} dB")
        print(f"Coverage : {len(valid)} / {acc.nrows * acc.ncols} cells "
              f"({100 * len(valid) / (acc.nrows * acc.ncols):.1f}%)")


if __name__ == "__main__":
    main()