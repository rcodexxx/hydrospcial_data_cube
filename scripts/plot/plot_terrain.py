"""
Generate MBES-derived terrain figures: VRM, Slope, BPI.

Bathymetry main figure is produced by plot_bathymetry.py (with
satellite basemap). This script handles the derived terrain layers
that share a flat-style presentation with bathymetric isobaths
overlaid for spatial reference.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_erosion

from src.config import ROOT, get_config


# Style constants
NODATA_FACE = "#cccccc"
ISOBATH_MAJOR_M = 10
ISOBATH_MINOR_M = 5
DEM_SMOOTH_SIGMA_PX = 10
TICK_FONTSIZE = 8


def read_masked(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        return data, src.bounds, src.crs


def smooth_dem(dem, sigma=DEM_SMOOTH_SIGMA_PX):
    mask = np.isfinite(dem)
    filled = np.where(mask, dem, 0.0)
    sm = gaussian_filter(filled, sigma=sigma)
    w = gaussian_filter(mask.astype(float), sigma=sigma)
    return np.where(w > 0.1, sm / np.maximum(w, 1e-10), np.nan)


def add_isobaths(ax, dem, extent, include_minor=False):
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
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)


def mask_to_dem(arr, dem):
    return np.where(np.isfinite(dem), arr, np.nan)


def plot_layer(arr, dem, bounds, crs, title, label,
               cmap, vmin, vmax, out_path, divergent=False):
    arr = mask_to_dem(arr, dem)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_facecolor(NODATA_FACE)

    im = ax.imshow(
        arr, extent=extent, origin="upper",
        cmap=cmap, vmin=vmin, vmax=vmax,
        aspect="equal", interpolation="bilinear",
    )

    add_isobaths(ax, dem, extent)

    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label(label, fontsize=10)

    # ax.set_title(title, fontsize=13)
    setup_geographic_axes(ax, bounds, crs)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    valid = arr[np.isfinite(arr)]
    if valid.size:
        print(f"  Saved: {out_path.name}  range "
              f"{valid.min():+.3f} ~ {valid.max():+.3f}, "
              f"std={valid.std():.3f}, clip [{vmin}, {vmax}]")


def main():
    cfg = get_config()
    mbes_tif = ROOT / cfg["mbes"]["bathymetry_tif"]
    vrm_tif = ROOT / cfg["mbes"]["vrm_tif"]
    slope_tif = ROOT / cfg["mbes"]["slope_tif"]
    bpi_tif = ROOT / cfg["mbes"]["bpi_tif"]

    out_dir = ROOT / "outputs/figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    dem, bounds, crs = read_masked(mbes_tif)
    vrm, _, _ = read_masked(vrm_tif)
    slope, _, _ = read_masked(slope_tif)
    bpi, _, _ = read_masked(bpi_tif)

    print("Generating terrain figures...")
    plot_layer(
        vrm, dem, bounds, crs,
        title="Mudan Reservoir — Vector Ruggedness Measure",
        label="VRM",
        cmap="viridis",
        vmin=0, vmax=float(np.nanpercentile(vrm, 98)),
        out_path=out_dir / "terrain_vrm.png",
    )
    plot_layer(
        slope, dem, bounds, crs,
        title="Mudan Reservoir — Slope",
        label="Slope (degree)",
        cmap="magma",
        vmin=0, vmax=float(np.nanpercentile(slope, 98)),
        out_path=out_dir / "terrain_slope.png",
    )

    bpi_clip = float(np.nanpercentile(np.abs(bpi), 98))
    plot_layer(
        bpi, dem, bounds, crs,
        title="Mudan Reservoir — Bathymetric Position Index",
        label="BPI (m)",
        cmap="RdBu_r",
        vmin=-bpi_clip, vmax=bpi_clip,
        out_path=out_dir / "terrain_bpi.png",
        divergent=True,
    )


if __name__ == "__main__":
    main()