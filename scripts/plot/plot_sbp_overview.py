# scripts/plot/plot_sub_bottom.py
"""
Generate sub-bottom layer figures for the report.
  1. Acoustic Impedance (Pa·s/m)
  2. Pulse Width at half maximum (m)
  3. Sediment Classification (Hamilton)
  4. Isopach - Sediment Thickness (m)
"""
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from pyproj import Transformer
from src.sbp.calculation import SEDIMENT_LABELS

ROOT      = Path(__file__).parent.parent.parent
MBES_TIF  = ROOT / "outputs/tif/mbes_bathymetry.tif"
Z_TIF     = ROOT / "outputs/tif/sbp_impedance.tif"
PW_TIF    = ROOT / "outputs/tif/sbp_pulse_width.tif"
SED_TIF   = ROOT / "outputs/tif/sbp_sediment_class.tif"
THICK_TIF = ROOT / "outputs/tif/sbp_isopach.tif"
OUT_DIR   = ROOT / "outputs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUT_NORTH = 2449050

SED_COLORS = [
    "#d73027",  # 0: Coarse sand
    "#f46d43",  # 1: Fine sand
    "#fdae61",  # 2: Very fine sand
    "#fee08b",  # 3: Silty sand
    "#ffffbf",  # 4: Sandy silt
    "#e6f598",  # 5: Silt
    "#d9ef8b",  # 6: Sandy-silt-clay
    "#a6d96a",  # 7: Silty clay
    "#66bd63",  # 8: Clayey silt
    "#1a9850",  # 9: Framework-supported mud
    "#4575b4",  # 10: Fluid mud
]


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
    z, _, _     = read_masked(Z_TIF, nodata=-9999.0)
    pw, _, _    = read_masked(PW_TIF, nodata=-9999.0)
    sed, _, _   = read_masked(SED_TIF, nodata=-1)
    thick, _, _ = read_masked(THICK_TIF, nodata=-9999.0)

    # apply cut and DEM mask to all layers
    dem_cut = dem.copy()
    cut_row = int((bounds.top - CUT_NORTH) / ((bounds.top - bounds.bottom) / dem.shape[0]))
    if cut_row > 0:
        dem_cut[:cut_row, :] = np.nan

    z     = apply_cut_and_mask(z, dem_cut, bounds)
    pw    = apply_cut_and_mask(pw, dem_cut, bounds)
    sed   = apply_cut_and_mask(sed, dem_cut, bounds)
    thick = apply_cut_and_mask(thick, dem_cut, bounds)

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    # # ── Figure 1: Acoustic Impedance ──────────────────────────
    # fig1, ax1 = plt.subplots(figsize=(10, 8))
    # z_lo, z_hi = np.nanpercentile(z[np.isfinite(z)], [2, 98])
    # im = ax1.imshow(z, extent=extent, origin="upper", cmap="turbo",
    #                 vmin=z_lo, vmax=z_hi, aspect="equal")
    # ax1.set_facecolor("#cccccc")
    # cb = plt.colorbar(im, ax=ax1, shrink=0.75, pad=0.02)
    # cb.set_label("Acoustic Impedance (Pa·s/m)", fontsize=10)
    # cb.formatter.set_scientific(True)
    # cb.formatter.set_powerlimits((6, 6))
    # cb.update_ticks()
    # ax1.set_title("Mudan Reservoir — Acoustic Impedance", fontsize=13)
    # setup_ax(ax1, bounds, crs)
    # plt.tight_layout()
    # plt.savefig(OUT_DIR / "sub_bottom_impedance.png", dpi=200, bbox_inches="tight")
    # print(f"Saved: {OUT_DIR / 'sub_bottom_impedance.png'}")

    # # ── Figure 2: Pulse Width ─────────────────────────────────
    # fig2, ax2 = plt.subplots(figsize=(10, 8))
    # pw_lo, pw_hi = np.nanpercentile(pw[np.isfinite(pw)], [2, 98])
    # im = ax2.imshow(pw, extent=extent, origin="upper", cmap="RdYlBu_r",
    #                 vmin=pw_lo, vmax=pw_hi, aspect="equal")
    # ax2.set_facecolor("#cccccc")
    # cb = plt.colorbar(im, ax=ax2, shrink=0.75, pad=0.02)
    # cb.set_label("Pulse Width at Half Maximum (m)", fontsize=10)
    # ax2.set_title("Mudan Reservoir — Seafloor Return Pulse Width", fontsize=13)
    # setup_ax(ax2, bounds, crs)
    # plt.tight_layout()
    # plt.savefig(OUT_DIR / "sub_bottom_pulse_width.png", dpi=200, bbox_inches="tight")
    # print(f"Saved: {OUT_DIR / 'sub_bottom_pulse_width.png'}")

    # # ── Figure 3: Sediment Classification ─────────────────────
    # fig3, ax3 = plt.subplots(figsize=(10, 8))
    # n_cls = len(SEDIMENT_LABELS)
    # cmap_sed = mcolors.ListedColormap(SED_COLORS)
    # bounds_sed = np.arange(-0.5, n_cls, 1)
    # norm_sed = mcolors.BoundaryNorm(bounds_sed, cmap_sed.N)

    # sed_m = np.ma.masked_where(~np.isfinite(sed) | (sed < 0), sed)
    # im = ax3.imshow(sed_m, extent=extent, origin="upper",
    #                 cmap=cmap_sed, norm=norm_sed, aspect="equal")
    # ax3.set_facecolor("#e0e0e0")
    # cbar = plt.colorbar(im, ax=ax3, ticks=range(n_cls), shrink=0.75, pad=0.02)
    # cbar.ax.set_yticklabels(SEDIMENT_LABELS, fontsize=7)
    # cbar.set_label("Sediment Type (coarse → fluid)", fontsize=10)
    # ax3.set_title("Mudan Reservoir — Sediment Classification", fontsize=13)
    # setup_ax(ax3, bounds, crs)
    # plt.tight_layout()
    # plt.savefig(OUT_DIR / "sub_bottom_sediment.png", dpi=200, bbox_inches="tight")
    # print(f"Saved: {OUT_DIR / 'sub_bottom_sediment.png'}")

    # ── Figure 4: Isopach ─────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(10, 8))

    thick = thick
    t_lo, t_hi = np.nanpercentile(thick[np.isfinite(thick)], [2, 98])

    im = ax4.imshow(thick, extent=extent, origin="upper", cmap="turbo",
                    vmin=t_lo, vmax=t_hi, aspect="equal")

    ax4.set_facecolor("#cccccc")
    cb = plt.colorbar(im, ax=ax4, shrink=0.75, pad=0.02)
    cb.set_label("Sediment Thickness (m)", fontsize=10)
    ax4.set_title("Mudan Reservoir — Isopach", fontsize=13)
    setup_ax(ax4, bounds, crs)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sub_bottom_isopach.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'sub_bottom_isopach.png'}")

    plt.show()

    # stats
    for name, arr in [("Impedance", z), ("Pulse Width", pw), ("Isopach", thick)]:
        v = arr[np.isfinite(arr)]
        if len(v):
            print(f"{name}: {v.min():.4f} ~ {v.max():.4f}, median={np.median(v):.4f}")

    sed_valid = sed[np.isfinite(sed) & (sed >= 0)]
    if len(sed_valid):
        print(f"\nSediment distribution:")
        total = len(sed_valid)
        for i, label in enumerate(SEDIMENT_LABELS):
            count = (sed_valid == i).sum()
            if count > 0:
                print(f"  {i:2d} {label:25s}: {count:7d} ({100 * count / total:.1f}%)")


if __name__ == "__main__":
    main()