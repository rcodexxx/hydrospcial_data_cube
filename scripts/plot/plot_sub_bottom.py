"""
Generate sub-bottom layer figures for the report.

Outputs:
  - sub_bottom_rl.png         continuous Reflection Loss with class boundaries
  - sub_bottom_sediment.png   discrete 7-class sediment classification
  - sub_bottom_isopach.png    sediment thickness
  - sub_bottom_confidence.png measurement coverage mask

Bathymetric isobaths (from MBES DEM) are overlaid on all substrate plots.
"""
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

from src.config import get_config, ROOT
from src.sbp.calculation import get_sediment_labels
from src.sbp.config import SEDIMENT_COLORS as SED_COLORS

# Class-boundary RL thresholds (Hamilton 1970), dB — upper bound per class.
CLASS_THRESHOLDS = [9.59, 12.98, 14.14, 19.18]

# Bathymetric contour settings
ISOBATH_MAJOR_M = 10        # labeled
ISOBATH_MINOR_M = 5         # unlabeled, fainter
DEM_SMOOTH_SIGMA_PX = 10    # suppresses per-cell DEM noise in contours

# Colorbar: aligned to main axis height
CBAR_WIDTH = "4%"
CBAR_PAD = 0.08


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
    data[~np.isfinite(dem)] = np.nan
    return data


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

    # 3 ticks instead of 4 to avoid label crowding
    ax.set_xticks(np.linspace(bounds.left, bounds.right, 4))
    ax.set_yticks(np.linspace(bounds.bottom, bounds.top, 4))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_lat))
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, color="white", linewidth=0.4, alpha=0.5, linestyle="--")


def add_colorbar(ax, im, label, ticks=None, ticklabels=None):
    """Append a colorbar to the right of `ax`, matched to its height."""
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size=CBAR_WIDTH, pad=CBAR_PAD)
    cbar = plt.colorbar(im, cax=cax, ticks=ticks)
    cbar.set_label(label, fontsize=10)
    if ticklabels is not None:
        cbar.ax.set_yticklabels(ticklabels, fontsize=8)
    return cbar


# ─────────────────────────── overlay helpers ──────────────────────
def smooth_dem(dem, sigma_px=DEM_SMOOTH_SIGMA_PX):
    """Gaussian-smooth DEM for clean isobath contours, preserving NaN."""
    mask = np.isfinite(dem)
    filled = np.where(mask, dem, 0)
    weights = mask.astype(np.float32)
    smoothed = gaussian_filter(filled, sigma=sigma_px)
    w_sm = gaussian_filter(weights, sigma=sigma_px)
    return np.where(w_sm > 0.1, smoothed / np.maximum(w_sm, 1e-10), np.nan)


def add_isobaths(ax, dem, extent, include_minor=True):
    """Overlay MBES bathymetric isobaths."""
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


# ─────────────────────────── individual plots ─────────────────────
def plot_rl(rl, dem, bounds, crs, out_path):
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    v = rl[np.isfinite(rl)]
    vmin, vmax = float(np.percentile(v, 2)), float(np.percentile(v, 98))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.ma.masked_invalid(rl),
                   extent=extent, origin="upper",
                   cmap="RdYlBu", vmin=vmin, vmax=vmax,
                   aspect="equal", interpolation="bilinear")
    ax.set_facecolor("#e0e0e0")

    add_isobaths(ax, dem, extent)
    add_colorbar(ax, im, "Reflection Loss (dB)\n← coarse  |  fluid →")
    setup_ax(ax, bounds, crs)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path.name}")
    plt.close()


def plot_sediment(sed, dem, bounds, crs, labels, out_path):
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    n_cls = len(labels)
    cmap = mcolors.ListedColormap(SED_COLORS[:n_cls])
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n_cls, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.ma.masked_invalid(sed),
                   extent=extent, origin="upper",
                   cmap=cmap, norm=norm,
                   aspect="equal", interpolation="nearest")
    ax.set_facecolor("#e0e0e0")

    add_isobaths(ax, dem, extent)
    add_colorbar(ax, im, "Sediment Type (coarse → fluid)",
                 ticks=range(n_cls), ticklabels=labels)
    setup_ax(ax, bounds, crs)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path.name}")
    plt.close()


def plot_isopach(thick, dem, bounds, crs, out_path):
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    v = thick[np.isfinite(thick)]
    lo, hi = np.percentile(v, [10, 90])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(thick, extent=extent, origin="upper",
                   cmap="turbo", vmin=lo, vmax=hi,
                   aspect="equal", interpolation="bilinear")
    ax.set_facecolor("#cccccc")

    add_isobaths(ax, dem, extent)
    add_colorbar(ax, im, "Sediment Thickness (m)")
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
                 ticks=[0, 1], ticklabels=["Measured", "IDW interpolated"])
    setup_ax(ax, bounds, crs)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path.name}")
    plt.close()


# ─────────────────────────── main ─────────────────────────────────
def main():
    cfg = get_config()
    mbes_tif = ROOT / cfg["mbes"]["bathymetry_tif"]
    rl_tif = ROOT / cfg["sbp"]["outputs"]["rl_tif"]
    sed_tif = ROOT / cfg["sbp"]["outputs"]["sediment_tif"]
    thick_tif = ROOT / cfg["sbp"]["outputs"]["isopach_tif"]
    conf_tif = ROOT / cfg["sbp"]["outputs"]["confidence_tif"]
    out_dir = ROOT / "outputs/figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = get_sediment_labels()

    dem, bounds, crs = read_masked(mbes_tif)
    rl, _, _ = read_masked(rl_tif, nodata=-9999.0)
    sed, _, _ = read_masked(sed_tif, nodata=-1)
    thick, _, _ = read_masked(thick_tif, nodata=-9999.0)
    conf, _, _ = read_masked(conf_tif, nodata=255)
    rl = apply_dem_mask(rl, dem)
    sed = apply_dem_mask(sed, dem)
    thick = apply_dem_mask(thick, dem)
    conf = apply_dem_mask(conf, dem)

    sed_valid = sed[np.isfinite(sed) & (sed >= 0)]
    total = len(sed_valid)
    print("Sediment distribution:")
    for i, label in enumerate(labels):
        pct = 100 * (sed_valid == i).sum() / max(total, 1)
        if pct > 0:
            print(f"  {i:2d} {label:30s}: {pct:5.1f}%")

    plot_rl(rl, dem, bounds, crs, out_dir / "sub_bottom_rl.png")
    plot_sediment(sed, dem, bounds, crs, labels,
                  out_dir / "sub_bottom_sediment.png")
    # plot_isopach(thick, dem, bounds, crs, out_dir / "sub_bottom_isopach.png")
    plot_confidence(conf, bounds, crs, out_dir / "sub_bottom_confidence.png")

    v = thick[np.isfinite(thick)]
    if len(v):
        print(f"\nIsopach: {v.min():.2f} ~ {v.max():.2f} m  "
              f"median={np.median(v):.2f} m")


if __name__ == "__main__":
    main()