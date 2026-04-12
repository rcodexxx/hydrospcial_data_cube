# scripts/plot/plot_terrain.py
"""
Generate terrain analysis figures for the report.
  1. Bathymetry with contour lines (turbo)
  2. VRM - Vector Ruggedness Measure (inferno)
  3. MBES Coverage / Confidence (binary valid mask)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).parent.parent.parent
DEM_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
VRM_TIF = ROOT / "outputs/tif/mbes_vrm.tif"
OUT_DIR = ROOT / "outputs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUT_NORTH = 2449050


def read_tif(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nd = src.nodata
        bounds = src.bounds
        crs = src.crs
        res = src.res[0]
    if nd is not None:
        data[data == nd] = np.nan
    return data, bounds, crs, res


def apply_cut(data, bounds, res):
    cut_row = int((bounds.top - CUT_NORTH) / res)
    if cut_row > 0:
        data[:cut_row, :] = np.nan
    return data


def setup_ax(ax, bounds, crs):
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    mid_x = (bounds.left + bounds.right) / 2
    mid_y = (bounds.bottom + bounds.top) / 2

    def fmt_lon(val, _):
        lon, _ = tr.transform(val, mid_y)
        d = int(abs(lon))
        m = (abs(lon) - d) * 60
        return f"{d}°{m:05.2f}′E"

    def fmt_lat(val, _):
        _, lat = tr.transform(mid_x, val)
        d = int(abs(lat))
        m = (abs(lat) - d) * 60
        return f"{d}°{m:05.2f}′N"

    ax.set_xticks(np.linspace(bounds.left, bounds.right, 4))
    ax.set_yticks(np.linspace(bounds.bottom, bounds.top, 4))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_lat))
    ax.tick_params(axis="x", labelsize=8, rotation=25)
    ax.tick_params(axis="y", labelsize=8)


def main():
    dem, bounds, crs, res = read_tif(DEM_TIF)
    vrm, _, _, _ = read_tif(VRM_TIF)

    dem = apply_cut(dem, bounds, res)
    vrm = apply_cut(vrm, bounds, res)

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    H, W = dem.shape

    # grid for contour
    xs = np.linspace(bounds.left, bounds.right, W)
    ys = np.linspace(bounds.top, bounds.bottom, H)
    X, Y = np.meshgrid(xs, ys)

    # ── Figure 1: Bathymetry + Contours ───────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 8))

    # filled background
    im = ax1.imshow(dem, extent=extent, origin="upper", cmap="turbo_r", aspect="equal")
    cb = plt.colorbar(im, ax=ax1, shrink=0.75, pad=0.02)
    cb.set_label("Depth (m, positive down)", fontsize=10)

    # contour lines
    dem_smooth = np.where(np.isfinite(dem), dem, np.nan)
    levels = np.arange(
        np.floor(np.nanmin(dem) / 5) * 5, np.ceil(np.nanmax(dem) / 5) * 5 + 1, 5
    )
    cs = ax1.contour(
        X, Y, dem_smooth, levels=levels, colors="black", linewidths=0.6, alpha=0.7
    )
    ax1.clabel(cs, fmt="%d m", fontsize=7, inline=True)

    ax1.set_title("Mudan Reservoir — Bathymetry", fontsize=13)
    ax1.set_facecolor("#cccccc")
    setup_ax(ax1, bounds, crs)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "terrain_bathymetry.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'terrain_bathymetry.png'}")

    # ── Figure 2: VRM ─────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 8))

    im = ax2.imshow(
        vrm,
        extent=extent,
        origin="upper",
        cmap="viridis",
        vmin=0,
        vmax=0.1,
        aspect="equal",
    )
    cb = plt.colorbar(im, ax=ax2, shrink=0.75, pad=0.02)
    cb.set_label("VRM", fontsize=10)

    ax2.set_title("Mudan Reservoir — Vector Ruggedness Measure", fontsize=13)
    ax2.set_facecolor("#cccccc")
    setup_ax(ax2, bounds, crs)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "terrain_vrm.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'terrain_vrm.png'}")

    plt.show()


if __name__ == "__main__":
    main()
