"""
Bathymetry figure with Esri World Imagery basemap and isobaths.
Style aligned with the web viewer (translucent overlay + contour lines).
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
import contextily as cx
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

from src.config import ROOT, get_config


# Style constants
CMAP = "turbo_r"
OVERLAY_ALPHA = 0.85
ISOBATH_MAJOR_M = 10
ISOBATH_MINOR_M = 5
DEM_SMOOTH_SIGMA_PX = 10
TICK_FONTSIZE = 8


def read_dem(path):
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


def add_isobaths(ax, dem, extent):
    dem_sm = np.ma.masked_invalid(smooth_dem(dem))
    depth_max = int(np.ceil(np.nanmax(dem_sm) / ISOBATH_MAJOR_M) * ISOBATH_MAJOR_M)

    minor = [d for d in range(ISOBATH_MINOR_M, depth_max, ISOBATH_MINOR_M)
             if d % ISOBATH_MAJOR_M != 0]
    if minor:
        ax.contour(dem_sm, levels=minor,
                   colors="black", linewidths=0.5, alpha=0.5,
                   extent=extent, origin="upper")

    major = list(range(ISOBATH_MAJOR_M, depth_max, ISOBATH_MAJOR_M))
    if major:
        cs = ax.contour(dem_sm, levels=major,
                        colors="black", linewidths=1, alpha=0.9,
                        extent=extent, origin="upper")
        ax.clabel(cs, inline=True, fontsize=7, fmt="%d m",
                  colors="black")


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


def main():
    cfg = get_config()
    mbes_tif = ROOT / cfg["mbes"]["bathymetry_tif"]
    out_dir = ROOT / "outputs/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bathymetry_overview.png"

    dem, bounds, crs = read_dem(mbes_tif)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    valid = dem[np.isfinite(dem)]
    vmin, vmax = float(np.percentile(valid, 1)), float(np.percentile(valid, 99))

    fig, ax = plt.subplots(figsize=(11, 8))

    # Translucent bathymetry overlay FIRST
    im = ax.imshow(
        np.ma.masked_invalid(dem),
        extent=extent, origin="upper",
        cmap=CMAP, vmin=vmin, vmax=vmax,
        alpha=OVERLAY_ALPHA, aspect="equal",
        interpolation="bilinear",
        zorder=2,
    )

    # Set extent before basemap call
    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)

    # Esri basemap UNDER overlay
    cx.add_basemap(
        ax,
        crs=crs,
        source=cx.providers.Esri.WorldImagery,
        zoom=17,
        attribution_size=6,
        zorder=1,
    )

    add_isobaths(ax, dem, extent)

    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Depth (m)", fontsize=10)

    # ax.set_title("Mudan Reservoir Bathymetry", fontsize=13)
    setup_geographic_axes(ax, bounds, crs)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path.relative_to(ROOT)}")
    print(f"  Depth range: {valid.min():.2f} ~ {valid.max():.2f} m")
    print(f"  Median: {np.median(valid):.2f} m")
    print(f"  Color clip: {vmin:.1f} ~ {vmax:.1f} m")


if __name__ == "__main__":
    main()