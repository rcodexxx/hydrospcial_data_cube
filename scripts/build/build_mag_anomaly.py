"""
Generate magnetometer anomaly figure with trackline overlay.

Background, residual, and confidence layers are computed in the
pipeline (build_mag_anomaly.py) but not visualized here:
  - Background field is uniform for Mudan Reservoir (σ = 3.6 nT)
  - Residual ≈ anomaly because background is negligible
  - Confidence is uniformly interpolated for dense surveys
The pipeline tools remain available for other sites.
"""
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

from src.config import get_config, ROOT
from src.mag.read_mag import read_mag


CMAP_DIVERGING = "RdBu_r"
COLOR_CLIP_PERCENTILE = 98
COLOR_ROUND_TO = 5
NODATA_FACE = "#cccccc"

ISOBATH_MAJOR_M = 10
DEM_SMOOTH_SIGMA_PX = 10
TICK_FONTSIZE = 8

TRACKLINE_COLOR = "#222222"
TRACKLINE_LINEWIDTH = 0.3
TRACKLINE_ALPHA = 0.4


def read_masked(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nd = src.nodata
        bounds = src.bounds
        crs = src.crs
    if nd is not None:
        data[data == nd] = np.nan
    return data, bounds, crs


def smooth_dem(dem, sigma=DEM_SMOOTH_SIGMA_PX):
    mask = np.isfinite(dem)
    filled = np.where(mask, dem, 0.0)
    sm = gaussian_filter(filled, sigma=sigma)
    w = gaussian_filter(mask.astype(float), sigma=sigma)
    return np.where(w > 0.1, sm / np.maximum(w, 1e-10), np.nan)


def add_isobaths(ax, dem, extent):
    if dem is None:
        return
    dem_sm = np.ma.masked_invalid(smooth_dem(dem))
    depth_max = int(np.ceil(np.nanmax(dem_sm) / ISOBATH_MAJOR_M) * ISOBATH_MAJOR_M)
    major = list(range(ISOBATH_MAJOR_M, depth_max, ISOBATH_MAJOR_M))
    if not major:
        return
    cs = ax.contour(dem_sm, levels=major,
                    colors="black", linewidths=0.5, alpha=0.6,
                    extent=extent, origin="upper")
    ax.clabel(cs, inline=True, fontsize=7, fmt="%d m")


def setup_geographic_axes(ax, bounds, crs):
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    mid_x = (bounds.left + bounds.right) / 2
    mid_y = (bounds.bottom + bounds.top) / 2

    def fmt_lon(val, _):
        lon, _ = tr.transform(val, mid_y)
        d, m = int(abs(lon)), (abs(lon) % 1) * 60
        return f"{d}°{m:05.2f}′E"

    def fmt_lat(val, _):
        _, lat = tr.transform(mid_x, val)
        d, m = int(abs(lat)), (abs(lat) % 1) * 60
        return f"{d}°{m:05.2f}′N"

    ax.set_xticks(np.linspace(bounds.left, bounds.right, 4))
    ax.set_yticks(np.linspace(bounds.bottom, bounds.top, 4))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_lat))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, rotation=25)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.grid(True, color="white", linewidth=0.4, alpha=0.5, linestyle="--")


def load_track_points(survey_dirs):
    all_x, all_y = [], []
    for entry in survey_dirs:
        mag_dir = ROOT / entry["path"]
        for f in sorted(mag_dir.glob("*.mag")):
            recs = read_mag(f, apply_layback=True)
            all_x.extend(r["x"] for r in recs)
            all_y.extend(r["y"] for r in recs)
    return np.array(all_x), np.array(all_y)


def main():
    cfg = get_config()
    mbes_tif = ROOT / cfg["mbes"]["bathymetry_tif"]
    an_tif = ROOT / cfg["mag"]["outputs"]["anomaly_tif"]

    out_dir = ROOT / "outputs/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mag_anomaly.png"

    dem, bounds, crs = read_masked(mbes_tif)
    an, _, _ = read_masked(an_tif)
    an = np.where(np.isfinite(dem), an, np.nan)

    valid = an[np.isfinite(an)]
    abs_clip = float(np.nanpercentile(np.abs(valid), COLOR_CLIP_PERCENTILE))
    vmax = max(np.ceil(abs_clip / COLOR_ROUND_TO) * COLOR_ROUND_TO, COLOR_ROUND_TO)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    print(f"Loading tracklines for overlay...")
    tx, ty = load_track_points(cfg["mag"]["survey_dirs"])

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_facecolor(NODATA_FACE)

    im = ax.imshow(an, extent=extent, origin="upper",
                   cmap=CMAP_DIVERGING, vmin=-vmax, vmax=vmax,
                   aspect="equal", interpolation="bilinear")
    add_isobaths(ax, dem, extent)

    ax.scatter(tx, ty, color=TRACKLINE_COLOR,
               s=TRACKLINE_LINEWIDTH, alpha=TRACKLINE_ALPHA,
               linewidths=0, zorder=2)

    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Magnetic Anomaly (nT)", fontsize=10)
    ax.set_title("Mudan Reservoir — Magnetic Anomaly", fontsize=13)
    setup_geographic_axes(ax, bounds, crs)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path.name}  range {valid.min():+.1f}~{valid.max():+.1f} nT, "
          f"std={valid.std():.1f} nT, clip ±{vmax:.0f} nT, n_tracks={len(tx)}")


if __name__ == "__main__":
    main()