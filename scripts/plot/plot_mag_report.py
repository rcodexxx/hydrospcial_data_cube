"""
Generate magnetometer figures: anomaly, background, residual, confidence.

Layers reuse SBP helper conventions:
  - smooth_dem + add_isobaths for bathymetric overlay
  - geographic axis formatter
  - percentile-clipped diverging colormap for nT fields
  - MBES mask (no manual cuts)
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

from src.config import get_config, ROOT


# ──────────────────────────────────────────────────────────
# Plot parameters (match SBP plot_sub_bottom.py style)
# ──────────────────────────────────────────────────────────
CMAP_DIVERGING = "RdBu_r"
COLOR_CLIP_PERCENTILE = 98       # symmetric clip of |value|
COLOR_ROUND_TO = 5               # round vmax to nearest 5 nT
NODATA_FACE = "#cccccc"

ISOBATH_MAJOR_M = 10
ISOBATH_MINOR_M = 5
DEM_SMOOTH_SIGMA_PX = 10
TICK_FONTSIZE = 8

# Confidence layer colors
CONF_COLORS = {
    0: "#2E7D32",   # measured (green)
    1: "#FFC107",   # interpolated (amber)
}
CONF_LABELS = {0: "Measured", 1: "Interpolated"}


def read_masked(path):
    with rasterio.open(path) as src:
        data = src.read(1)
        nd = src.nodata
        bounds = src.bounds
        crs = src.crs
    if data.dtype.kind == "f":
        data = data.astype(np.float32)
        if nd is not None:
            data[data == nd] = np.nan
    return data, bounds, crs, nd


def smooth_dem(dem, sigma=DEM_SMOOTH_SIGMA_PX):
    """NaN-aware Gaussian smoothing for isobath cleanliness."""
    mask = np.isfinite(dem)
    filled = np.where(mask, dem, 0.0)
    sm = gaussian_filter(filled, sigma=sigma)
    w = gaussian_filter(mask.astype(float), sigma=sigma)
    return np.where(w > 0.1, sm / np.maximum(w, 1e-10), np.nan)


def add_isobaths(ax, dem, extent, include_minor=False):
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


def setup_geographic_axes(ax, bounds, crs):
    """Format axes as degree-minute geographic labels."""
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


def mask_to_valid_dem(arr, dem):
    """Set values to NaN where DEM is NaN (outside MBES coverage)."""
    return np.where(np.isfinite(dem), arr, np.nan)


def plot_diverging_field(arr, dem, bounds, crs, title, label, out_path):
    """Plotter for anomaly / background / residual (nT, diverging cmap)."""
    arr = mask_to_valid_dem(arr, dem)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        print(f"  Warning: no valid data in {title}")
        return

    abs_clip = float(np.nanpercentile(np.abs(valid), COLOR_CLIP_PERCENTILE))
    vmax = max(np.ceil(abs_clip / COLOR_ROUND_TO) * COLOR_ROUND_TO, COLOR_ROUND_TO)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_facecolor(NODATA_FACE)

    im = ax.imshow(
        arr, extent=extent, origin="upper",
        cmap=CMAP_DIVERGING, vmin=-vmax, vmax=vmax,
        aspect="equal", interpolation="bilinear",
    )
    add_isobaths(ax, dem, extent, include_minor=False)

    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label(label, fontsize=10)

    ax.set_title(title, fontsize=13)
    setup_geographic_axes(ax, bounds, crs)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {out_path.name}  range {valid.min():+.1f}~{valid.max():+.1f} nT, "
          f"std={valid.std():.1f} nT, clip ±{vmax:.0f} nT")


def plot_confidence(conf, dem, bounds, crs, title, out_path):
    """Plotter for confidence layer (categorical: measured vs interpolated)."""
    # Mask to MBES coverage; nodata stays as 255
    conf = conf.copy()
    conf_masked = np.where(np.isfinite(dem), conf, 255)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    cmap = ListedColormap([CONF_COLORS[0], CONF_COLORS[1]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    display = np.where(conf_masked == 255, np.nan,
                       conf_masked.astype(np.float32))

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_facecolor(NODATA_FACE)

    ax.imshow(
        display, extent=extent, origin="upper",
        cmap=cmap, norm=norm,
        aspect="equal", interpolation="nearest",
    )
    add_isobaths(ax, dem, extent, include_minor=False)

    # Custom legend instead of colorbar (categorical data)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CONF_COLORS[k])
               for k in [0, 1]]
    labels = [CONF_LABELS[0], CONF_LABELS[1]]
    ax.legend(handles, labels, loc="upper right", framealpha=0.9, fontsize=9)

    ax.set_title(title, fontsize=13)
    setup_geographic_axes(ax, bounds, crs)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    n_meas = ((conf_masked == 0)).sum()
    n_int = ((conf_masked == 1)).sum()
    total = n_meas + n_int
    print(f"  Saved: {out_path.name}  "
          f"measured={n_meas} ({100*n_meas/max(total,1):.1f}%), "
          f"interpolated={n_int} ({100*n_int/max(total,1):.1f}%)")


def main():
    cfg = get_config()
    mbes_tif = ROOT / cfg["mbes"]["bathymetry_tif"]
    an_tif = ROOT / cfg["mag"]["outputs"]["anomaly_tif"]
    bg_tif = ROOT / cfg["mag"]["outputs"]["background_tif"]
    re_tif = ROOT / cfg["mag"]["outputs"]["residual_tif"]
    cf_tif = ROOT / cfg["mag"]["outputs"]["confidence_tif"]

    out_dir = ROOT / "outputs/figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    dem, bounds, crs, _ = read_masked(mbes_tif)
    an, _, _, _ = read_masked(an_tif)
    bg, _, _, _ = read_masked(bg_tif)
    re, _, _, _ = read_masked(re_tif)
    cf, _, _, _ = read_masked(cf_tif)

    print("Generating magnetometer figures...")
    plot_diverging_field(
        an, dem, bounds, crs,
        title="Mudan Reservoir — Magnetic Anomaly (F − IGRF)",
        label="Magnetic Anomaly (nT)",
        out_path=out_dir / "mag_anomaly.png",
    )
    plot_diverging_field(
        bg, dem, bounds, crs,
        title="Mudan Reservoir — Magnetic Background Field",
        label="Magnetic Background (nT)",
        out_path=out_dir / "mag_background.png",
    )
    plot_diverging_field(
        re, dem, bounds, crs,
        title="Mudan Reservoir — Magnetic Residual (Local Anomaly)",
        label="Magnetic Residual (nT)",
        out_path=out_dir / "mag_residual.png",
    )
    plot_confidence(
        cf, dem, bounds, crs,
        title="Mudan Reservoir — Magnetic Data Confidence",
        out_path=out_dir / "mag_confidence.png",
    )


if __name__ == "__main__":
    main()