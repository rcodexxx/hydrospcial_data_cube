"""
Generate magnetic layer figures for the report.

Outputs:
  - mag_anomaly.png       Total field anomaly (F - IGRF), divergent
  - mag_residual.png      Anomaly minus smoothed background, divergent
  - mag_confidence.png    Measurement coverage mask

Both anomaly and residual figures overlay detected targets when the
target CSV is available.
"""
import csv

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

from src.config import get_config, ROOT


# Bathymetric contour settings (match plot_sub_bottom.py)
ISOBATH_MAJOR_M = 10
ISOBATH_MINOR_M = 5
DEM_SMOOTH_SIGMA_PX = 10

# Colorbar geometry
CBAR_WIDTH = "4%"
CBAR_PAD = 0.08

# Target marker style
TARGET_MARKER_EDGE = "white"
TARGET_MARKER_FACE = "none"
TARGET_MARKER_LW = 1.2


# ─────────────────────────── I/O helpers ───────────────────────────
def read_masked(path, nodata=None):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nd = nodata if nodata is not None else src.nodata
        bounds = src.bounds
        crs = src.crs
    if nd is not None:
        data[data == nd] = np.nan
    return data, bounds, crs


def apply_dem_mask(data, dem):
    data = data.copy()
    data[~np.isfinite(dem)] = np.nan
    return data


def read_targets_csv(path):
    if not path.exists():
        return np.array([]), np.array([]), np.array([])
    xs, ys, vs = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x_m"]))
            ys.append(float(row["y_m"]))
            vs.append(float(row["anomaly_nT"]))
    return np.array(xs), np.array(ys), np.array(vs)


# ─────────────────────────── axis formatting ──────────────────────
def setup_ax(ax, bounds, crs):
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
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, color="white", linewidth=0.4, alpha=0.5, linestyle="--")


def add_colorbar(ax, im, label, ticks=None, ticklabels=None):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size=CBAR_WIDTH, pad=CBAR_PAD)
    cbar = plt.colorbar(im, cax=cax, ticks=ticks)
    cbar.set_label(label, fontsize=10)
    if ticklabels is not None:
        cbar.ax.set_yticklabels(ticklabels, fontsize=8)
    return cbar


# ─────────────────────────── overlay helpers ──────────────────────
def smooth_dem(dem, sigma_px=DEM_SMOOTH_SIGMA_PX):
    mask = np.isfinite(dem)
    filled = np.where(mask, dem, 0)
    weights = mask.astype(np.float32)
    smoothed = gaussian_filter(filled, sigma=sigma_px)
    w_sm = gaussian_filter(weights, sigma=sigma_px)
    return np.where(w_sm > 0.1, smoothed / np.maximum(w_sm, 1e-10), np.nan)


def add_isobaths(ax, dem, extent, include_minor=True):
    if dem is None:
        return
    dem_sm = np.ma.masked_invalid(smooth_dem(dem))
    depth_max = int(np.ceil(np.nanmax(dem_sm) / ISOBATH_MAJOR_M) * ISOBATH_MAJOR_M)

    if include_minor:
        minor = [d for d in range(ISOBATH_MINOR_M, depth_max, ISOBATH_MINOR_M)
                 if d % ISOBATH_MAJOR_M != 0]
        if minor:
            ax.contour(dem_sm, levels=minor,
                       colors="black", linewidths=0.2, alpha=0.25,
                       extent=extent, origin="upper")

    major = list(range(ISOBATH_MAJOR_M, depth_max, ISOBATH_MAJOR_M))
    if major:
        cs = ax.contour(dem_sm, levels=major,
                        colors="black", linewidths=0.5, alpha=0.6,
                        extent=extent, origin="upper")
        ax.clabel(cs, inline=True, fontsize=7, fmt="%d m")


def overlay_targets(ax, xs, ys, vals=None):
    if len(xs) == 0:
        return
    if vals is not None and len(vals) > 0:
        avals = np.abs(vals)
        sizes = np.clip(20 + 15 * np.log10(np.maximum(avals, 10) / 10),
                        20, 100)
    else:
        sizes = 35
    ax.scatter(xs, ys, s=sizes,
               facecolors=TARGET_MARKER_FACE,
               edgecolors=TARGET_MARKER_EDGE,
               linewidths=TARGET_MARKER_LW,
               zorder=10)


# ─────────────────────────── individual plots ─────────────────────
def plot_signed_field(arr, dem, bounds, crs, label, out_path,
                      targets_csv=None):
    """Plot signed magnetic field (anomaly or residual) with diverging colormap."""
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    v = arr[np.isfinite(arr)]
    lo, hi = np.percentile(v, [2, 98])
    bound = max(abs(lo), abs(hi))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.ma.masked_invalid(arr),
                   extent=extent, origin="upper",
                   cmap="RdBu_r", vmin=-bound, vmax=bound,
                   aspect="equal", interpolation="bilinear")
    ax.set_facecolor("#e0e0e0")

    add_isobaths(ax, dem, extent)

    if targets_csv is not None and targets_csv.exists():
        xs, ys, vals = read_targets_csv(targets_csv)
        overlay_targets(ax, xs, ys, vals)
        if len(xs):
            ax.text(0.02, 0.02, f"{len(xs)} targets",
                    transform=ax.transAxes, fontsize=9,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.5,
                              edgecolor="none", pad=3))

    add_colorbar(ax, im, label)
    setup_ax(ax, bounds, crs)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path.name}")
    plt.close()


def plot_confidence(conf, bounds, crs, out_path):
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    conf_vis = np.where(conf == 255, np.nan, conf).astype(np.float32)
    cmap = mcolors.ListedColormap(["#1a9850", "#fee08b"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.ma.masked_invalid(conf_vis),
                   extent=extent, origin="upper",
                   cmap=cmap, norm=norm, aspect="equal")
    ax.set_facecolor("#e0e0e0")

    add_colorbar(ax, im, "Data confidence",
                 ticks=[0, 1], ticklabels=["Measured", "Interpolated"])
    setup_ax(ax, bounds, crs)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path.name}")
    plt.close()


# ─────────────────────────── entry ────────────────────────────────
def main():
    cfg = get_config()
    mag_cfg = cfg["mag"]
    mbes_tif = ROOT / cfg["mbes"]["bathymetry_tif"]
    anom_tif = ROOT / mag_cfg["outputs"]["anomaly_tif"]
    res_tif = ROOT / mag_cfg["outputs"]["residual_tif"]
    conf_tif = ROOT / mag_cfg["outputs"]["confidence_tif"]
    targets_csv = ROOT / mag_cfg["outputs"]["targets_csv"]

    out_dir = ROOT / "outputs/figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    dem, _, _ = read_masked(mbes_tif)

    if anom_tif.exists():
        anom, bounds, crs = read_masked(anom_tif)
        anom = apply_dem_mask(anom, dem)
        plot_signed_field(anom, dem, bounds, crs,
                          "Magnetic anomaly (nT)\nF − IGRF",
                          out_dir / "mag_anomaly.png",
                          targets_csv=targets_csv)
    else:
        print(f"Skip mag_anomaly.png : {anom_tif.name} not found")

    if res_tif.exists():
        res, bounds, crs = read_masked(res_tif)
        res = apply_dem_mask(res, dem)
        plot_signed_field(res, dem, bounds, crs,
                          "Residual anomaly (nT)\nanomaly − background",
                          out_dir / "mag_residual.png",
                          targets_csv=targets_csv)
    else:
        print(f"Skip mag_residual.png : {res_tif.name} not found")

    if conf_tif.exists():
        with rasterio.open(conf_tif) as src:
            conf = src.read(1)
            bounds = src.bounds
            crs = src.crs
        plot_confidence(conf, bounds, crs, out_dir / "mag_confidence.png")
    else:
        print(f"Skip mag_confidence.png : {conf_tif.name} not found")


if __name__ == "__main__":
    main()