# scripts/plot/plot_sbp.py
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).parent.parent.parent
MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
RL_TIF = ROOT / "outputs/tif/sbp_rl.tif"
SED_TIF = ROOT / "outputs/tif/sbp_sediment_class.tif"
CONF_TIF = ROOT / "outputs/tif/sbp_confidence.tif"
OUT_DIR = ROOT / "outputs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUT_NORTH = 2449050

SEDIMENT_LABELS = [
    "Coarse sand",
    "Fine sand",
    "Very fine sand",
    "Silty sand",
    "Sandy silt",
    "Silt",
    "Sandy-silt-clay",
    "Silty clay",
    "Clayey silt",
    "Framework-supported mud",
    "Fluid mud"
]

# 各分類的上限閾值 (對應 SEDIMENT_LABELS 的前 10 項)
SEDIMENT_THRESHOLDS = [7.33, 8.02, 8.73, 9.63, 9.82, 10.25, 11.98, 13.20, 13.37, 22.40]

# sequential colormap: warm (coarse) → cool (fine) → blue (fluid)
# 擴充為 11 個顏色
SED_COLORS = [
    "#d73027",  # 0: Coarse sand - red
    "#f46d43",  # 1: Fine sand
    "#fdae61",  # 2: Very fine sand
    "#fee08b",  # 3: Silty sand
    "#ffffbf",  # 4: Sandy silt
    "#e6f598",  # 5: Silt - pale yellow-green
    "#d9ef8b",  # 6: Sandy-silt-clay
    "#a6d96a",  # 7: Silty clay - light green
    "#66bd63",  # 8: Clayey silt - green
    "#1a9850",  # 9: Framework-supported mud - dark green
    "#4575b4",  # 10: Fluid mud - deep blue
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


def apply_cut(data, bounds):
    rows = data.shape[0]
    res = (bounds.top - bounds.bottom) / rows
    cut_row = int((bounds.top - CUT_NORTH) / res)
    mask = np.ones_like(data, dtype=bool)
    mask[:cut_row, :] = False
    data[~mask] = np.nan
    return data


def setup_ax(ax, bounds, crs):
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    mid_x = (bounds.left + bounds.right) / 2
    mid_y = (bounds.bottom + bounds.top) / 2

    def fmt_lon(val, _):
        lon, _ = tr.transform(val, mid_y)
        d, m = int(abs(lon)), (abs(lon) % 1) * 60
        return f"{d}°{m:05.2f}′{'E' if lon >= 0 else 'W'}"

    def fmt_lat(val, _):
        _, lat = tr.transform(mid_x, val)
        d, m = int(abs(lat)), (abs(lat) % 1) * 60
        return f"{d}°{m:05.2f}′{'N' if lat >= 0 else 'S'}"

    ax.set_xticks(np.linspace(bounds.left, bounds.right, 4))
    ax.set_yticks(np.linspace(bounds.bottom, bounds.top, 4))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_lat))
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, color="white", linewidth=0.4, alpha=0.5, linestyle="--")


def main():
    rl, bounds, crs = read_masked(RL_TIF, nodata=-9999.0)
    sed, _, _ = read_masked(SED_TIF, nodata=-1)
    conf, _, _ = read_masked(CONF_TIF, nodata=255)
    dem, _, _ = read_masked(MBES_TIF)

    # apply north cut + DEM mask
    dem = apply_cut(dem, bounds)
    valid = np.isfinite(dem)
    rl[~valid] = np.nan
    sed[~valid] = np.nan
    conf[~valid] = np.nan

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    # confidence stats
    conf_flat = conf[np.isfinite(conf)]
    n_measured = (conf_flat == 0).sum()
    n_predicted = (conf_flat == 1).sum()
    n_total = n_measured + n_predicted
    print(f"Confidence: measured={n_measured} ({100 * n_measured / n_total:.1f}%), "
          f"predicted={n_predicted} ({100 * n_predicted / n_total:.1f}%)")

    # ── Figure 1: RL map ──────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 7))
    rl_lo, rl_hi = np.nanpercentile(rl[np.isfinite(rl)], [2, 98])
    im = ax1.imshow(rl, extent=extent, origin="upper", cmap="turbo",
                    vmin=rl_lo, vmax=rl_hi, aspect="equal")
    plt.colorbar(im, ax=ax1, label="RL (dB)", shrink=0.8)
    ax1.set_title("SBP Reflection Loss", fontsize=12)
    setup_ax(ax1, bounds, crs)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sbp_rl_map.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'sbp_rl_map.png'}")

    # ── Figure 2: Sediment classification ─────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    n_cls = len(SEDIMENT_LABELS)
    cmap_sed = mcolors.ListedColormap(SED_COLORS)
    bounds_sed = np.arange(-0.5, n_cls, 1)
    norm_sed = mcolors.BoundaryNorm(bounds_sed, cmap_sed.N)

    sed_m = np.ma.masked_where(~np.isfinite(sed) | (sed < 0), sed)
    im = ax2.imshow(sed_m, extent=extent, origin="upper",
                    cmap=cmap_sed, norm=norm_sed, aspect="equal")
    ax2.set_facecolor("#e0e0e0")
    cbar = plt.colorbar(im, ax=ax2, ticks=range(n_cls), shrink=0.8)
    cbar.ax.set_yticklabels(SEDIMENT_LABELS, fontsize=8)
    cbar.set_label("Sediment Type (coarse → fluid)", fontsize=9)
    ax2.set_title("Sediment Classification (Hamilton & Wood Models)", fontsize=12)
    setup_ax(ax2, bounds, crs)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sbp_sediment_map.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'sbp_sediment_map.png'}")

    # ── Figure 3: RL histogram ────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    rl_valid = rl[np.isfinite(rl)].ravel()
    ax3.hist(rl_valid, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    ax3.set_xlabel("Reflection Loss (dB)", fontsize=11)
    ax3.set_ylabel("Pixel Count", fontsize=11)
    ax3.set_title("RL Distribution with Physical Boundaries", fontsize=12)

    # Reference lines for each threshold
    for i, (label, rl_ref) in enumerate(zip(SEDIMENT_LABELS[:-1], SEDIMENT_THRESHOLDS)):
        # 標示 Hamilton 與 Wood 的物理極限
        is_major_boundary = (i == 7) or (i == 8)
        line_color = "red" if is_major_boundary else "orange"
        line_alpha = 0.9 if is_major_boundary else 0.6
        line_width = 1.2 if is_major_boundary else 0.8

        ax3.axvline(rl_ref, color=line_color, linewidth=line_width, alpha=line_alpha, linestyle="--")
        ax3.text(rl_ref + 0.15, ax3.get_ylim()[1] * 0.92,
                 label, fontsize=7, color=line_color, rotation=90, va="top")

    # 補上最後一個 Fluid mud 的文字標示
    ax3.text(SEDIMENT_THRESHOLDS[-1] + 0.5, ax3.get_ylim()[1] * 0.5,
             "Fluid mud\n(> 22.4 dB)", fontsize=9, color="blue", fontweight="bold", va="center")

    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sbp_rl_histogram.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'sbp_rl_histogram.png'}")

    plt.show()


if __name__ == "__main__":
    main()