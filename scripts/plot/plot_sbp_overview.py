# scripts/plot/plot_sub_bottom.py
"""
Generate sub-bottom layer figures for the report.
  1. Sediment Classification (Hamilton)
  2. Isopach - Sediment Thickness (m)
"""
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from pathlib import Path
from pyproj import Transformer
from src.sbp.calculation import SEDIMENT_LABELS

ROOT     = Path(__file__).parent.parent.parent
MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
SED_TIF  = ROOT / "outputs/tif/sbp_sediment_class.tif"
THICK_TIF= ROOT / "outputs/tif/sbp_isopach.tif"
OUT_DIR  = ROOT / "outputs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SED_COLORS = [
    "#d73027",  # 0: Coarse sand
    "#f46d43",  # 1: Fine sand
    "#fdae61",  # 2: Very fine sand
    "#fee08b",  # 3: Silty sand
    "#ffffbf",  # 4: Sandy silt
    "#e6f598",  # 5: Silt
    "#d9ef8b",  # 6: Sandy-silt-clay
    "#a6d96a",  # 7: Silty clay, Clayey silt
    "#66bd63",  # 8: Compacted mud
    "#4575b4",  # 9: Fluid mud
]


def read_masked(path, nodata=None):
    with rasterio.open(path) as src:
        data   = src.read(1).astype(np.float32)
        nd     = nodata if nodata is not None else src.nodata
        bounds = src.bounds
        crs    = src.crs
    if nd is not None:
        data[data == nd] = np.nan
    return data, bounds, crs


def apply_dem_mask(data, dem):
    data[~np.isfinite(dem)] = np.nan
    return data


def setup_ax(ax, bounds, crs):
    tr    = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    mid_x = (bounds.left  + bounds.right) / 2
    mid_y = (bounds.bottom + bounds.top)  / 2

    def fmt_lon(val, _):
        lon, _ = tr.transform(val, mid_y)
        d, m   = int(abs(lon)), (abs(lon) % 1) * 60
        return f"{d}°{m:05.2f}′E"

    def fmt_lat(val, _):
        _, lat = tr.transform(mid_x, val)
        d, m   = int(abs(lat)), (abs(lat) % 1) * 60
        return f"{d}°{m:05.2f}′N"

    ax.set_xticks(np.linspace(bounds.left,   bounds.right, 4))
    ax.set_yticks(np.linspace(bounds.bottom, bounds.top,   4))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_lat))
    ax.tick_params(axis="x", labelsize=8, rotation=25)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, color="white", linewidth=0.4, alpha=0.5, linestyle="--")


def main():
    dem, bounds, crs = read_masked(MBES_TIF)

    sed,   _, _ = read_masked(SED_TIF,   nodata=-1)
    thick, _, _ = read_masked(THICK_TIF, nodata=-9999.0)

    sed   = apply_dem_mask(sed,   dem)
    thick = apply_dem_mask(thick, dem)

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    # ── 計算各類別比例 ────────────────────────────────────────
    n_cls     = len(SEDIMENT_LABELS)
    sed_valid = sed[np.isfinite(sed) & (sed >= 0)]
    total     = len(sed_valid)
    pcts      = {i: 100 * (sed_valid == i).sum() / max(total, 1)
                 for i in range(n_cls)}

    print("Sediment distribution:")
    for i, label in enumerate(SEDIMENT_LABELS):
        if pcts[i] > 0:
            print(f"  {i:2d} {label:25s}: {pcts[i]:5.1f}%")

    # ── Figure 1: Sediment Classification ────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap_sed = mcolors.ListedColormap(SED_COLORS[:n_cls])

    # remap all valid class IDs to 0..n_cls-1
    sed_remapped = np.full_like(sed, np.nan)
    for i in range(n_cls):
        sed_remapped[sed == i] = i

    bounds_sed = np.arange(-0.5, n_cls, 1)
    norm_sed   = mcolors.BoundaryNorm(bounds_sed, cmap_sed.N)

    sed_m = np.ma.masked_invalid(sed_remapped)
    im = ax.imshow(sed_m, extent=extent, origin="upper",
                   cmap=cmap_sed, norm=norm_sed, aspect="equal")
    ax.set_facecolor("#e0e0e0")

    cbar = plt.colorbar(im, ax=ax, ticks=range(n_cls),
                        shrink=0.75, pad=0.02)
    cbar.ax.set_yticklabels(SEDIMENT_LABELS, fontsize=8)
    cbar.set_label("Sediment Type (coarse → fluid)", fontsize=10)
    ax.set_title("Mudan Reservoir — Sediment Classification", fontsize=13)
    setup_ax(ax, bounds, crs)
    plt.tight_layout()
    out = OUT_DIR / "sub_bottom_sediment.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()

    # ── Figure 2: Isopach ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))

    t_lo, t_hi = np.nanpercentile(thick[np.isfinite(thick)], [2, 98])
    im = ax.imshow(thick, extent=extent, origin="upper",
                   cmap="turbo", vmin=t_lo, vmax=t_hi, aspect="equal")
    ax.set_facecolor("#cccccc")
    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Sediment Thickness (m)", fontsize=10)
    ax.set_title("Mudan Reservoir — Isopach", fontsize=13)
    setup_ax(ax, bounds, crs)
    plt.tight_layout()
    out = OUT_DIR / "sub_bottom_isopach.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # ── Stats ─────────────────────────────────────────────────
    v = thick[np.isfinite(thick)]
    if len(v):
        print(f"\nIsopach: {v.min():.2f} ~ {v.max():.2f} m  "
              f"median={np.median(v):.2f} m")


if __name__ == "__main__":
    main()