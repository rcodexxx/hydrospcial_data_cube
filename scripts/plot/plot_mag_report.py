# scripts/plot/plot_mag.py
"""
Generate magnetometer figures for the report.
  1. Magnetic Background (large-scale geological trend)
  2. Magnetic Residual (local anomalies for UCH detection)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).parent.parent.parent
MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
BG_TIF = ROOT / "outputs/tif/mag_background.tif"
RES_TIF = ROOT / "outputs/tif/mag_residual.tif"
OUT_DIR = ROOT / "outputs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUT_NORTH = 2449050


def read_masked(path, nodata=None):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nd = nodata if nodata is not None else src.nodata
        bounds = src.bounds
        crs = src.crs
    if nd is not None:
        data[data == nd] = np.nan
    return data, bounds, crs


def apply_cut_and_mask(data, dem, bounds):
    rows = data.shape[0]
    res = (bounds.top - bounds.bottom) / rows
    cut_row = int((bounds.top - CUT_NORTH) / res)
    if cut_row > 0:
        data[:cut_row, :] = np.nan
    data[~np.isfinite(dem)] = np.nan
    return data


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
    ax.tick_params(axis="x", labelsize=8, rotation=25)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, color="white", linewidth=0.4, alpha=0.5, linestyle="--")


def main():
    dem, bounds, crs = read_masked(MBES_TIF)
    bg, _, _ = read_masked(BG_TIF, nodata=-9999.0)
    resid, _, _ = read_masked(RES_TIF, nodata=-9999.0)

    # apply cut
    dem_cut = dem.copy()
    cut_row = int(
        (bounds.top - CUT_NORTH) / ((bounds.top - bounds.bottom) / dem.shape[0])
    )
    if cut_row > 0:
        dem_cut[:cut_row, :] = np.nan

    bg = apply_cut_and_mask(bg, dem_cut, bounds)
    resid = apply_cut_and_mask(resid, dem_cut, bounds)

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    # ── Figure 1: Magnetic Background ─────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 8))

    bg_valid = bg[np.isfinite(bg)]
    bg_abs = float(np.nanpercentile(np.abs(bg_valid), 98))
    vmax_bg = np.ceil(bg_abs / 5) * 5

    im = ax1.imshow(
        bg,
        extent=extent,
        origin="upper",
        cmap="RdBu_r",
        vmin=-vmax_bg,
        vmax=vmax_bg,
        aspect="equal",
    )
    ax1.set_facecolor("#cccccc")
    cb = plt.colorbar(im, ax=ax1, shrink=0.75, pad=0.02)
    cb.set_label("Magnetic Background (nT)", fontsize=10)
    ax1.set_title("Mudan Reservoir — Magnetic Background Field", fontsize=13)
    setup_ax(ax1, bounds, crs)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mag_background.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'mag_background.png'}")

    # ── Figure 2: Magnetic Residual ───────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 8))

    res_valid = resid[np.isfinite(resid)]
    res_abs = float(np.nanpercentile(np.abs(res_valid), 98))
    vmax_res = np.ceil(res_abs / 5) * 5

    im = ax2.imshow(
        resid,
        extent=extent,
        origin="upper",
        cmap="RdBu_r",
        vmin=-vmax_res,
        vmax=vmax_res,
        aspect="equal",
    )
    ax2.set_facecolor("#cccccc")
    cb = plt.colorbar(im, ax=ax2, shrink=0.75, pad=0.02)
    cb.set_label("Magnetic Residual (nT)", fontsize=10)
    ax2.set_title("Mudan Reservoir — Magnetic Residual (Local Anomaly)", fontsize=13)
    setup_ax(ax2, bounds, crs)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mag_residual.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'mag_residual.png'}")

    plt.show()

    # stats
    print(
        f"\nBackground: {bg_valid.min():.1f} ~ {bg_valid.max():.1f} nT, "
        f"std={bg_valid.std():.1f} nT"
    )
    print(
        f"Residual:   {res_valid.min():.1f} ~ {res_valid.max():.1f} nT, "
        f"std={res_valid.std():.1f} nT"
    )


if __name__ == "__main__":
    main()
