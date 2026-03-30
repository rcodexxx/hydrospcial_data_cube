# scripts/plot/plot_isopach.py
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from pyproj import Transformer

ROOT      = Path(__file__).parent.parent.parent
MBES_TIF  = ROOT / "outputs/tif/mbes_bathymetry.tif"
THICK_TIF = ROOT / "outputs/tif/sbp_isopach.tif"
CONF_TIF  = ROOT / "outputs/tif/sbp_isopach_confidence.tif"
OUT_DIR   = ROOT / "outputs/figures"
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


def apply_cut(data, bounds):
    rows = data.shape[0]
    res = (bounds.top - bounds.bottom) / rows
    cut_row = int((bounds.top - CUT_NORTH) / res)
    data[:cut_row, :] = np.nan
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
    dem, bounds, crs = read_masked(MBES_TIF)
    thick, _, _ = read_masked(THICK_TIF, nodata=-9999.0)
    conf, _, _ = read_masked(CONF_TIF, nodata=0)

    # apply north cut + DEM mask
    dem = apply_cut(dem, bounds)
    valid = np.isfinite(dem)
    thick[~valid] = np.nan
    conf[~valid] = np.nan

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    # ── Figure 1: Isopach map ─────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 7))
    im = ax1.imshow(thick, extent=extent, origin="upper", cmap="YlOrRd",
                    vmin=0.5, vmax=0.9, aspect="equal")
    ax1.set_facecolor("#cccccc")
    plt.colorbar(im, ax=ax1, label="Sediment Thickness (m)", shrink=0.8)
    ax1.set_title("Isopach Map (SBP Sediment Thickness)", fontsize=12)
    setup_ax(ax1, bounds, crs)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "isopach_map.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'isopach_map.png'}")

    # ── Figure 2: Confidence map ──────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    conf_display = np.ma.masked_where(~np.isfinite(conf) | (conf == 0), conf)
    im = ax2.imshow(conf_display, extent=extent, origin="upper",
                    cmap="RdYlGn", vmin=1, vmax=255, aspect="equal")
    ax2.set_facecolor("#cccccc")
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label("Confidence (higher = better)", fontsize=9)
    cbar.set_ticks([1, 64, 128, 192, 255])
    cbar.set_ticklabels(["Low\n(far from\ntrack)", "", "Track\nedge", "",
                         "High\n(on track,\nhigh SNR)"])
    ax2.set_title("Isopach Confidence Mask", fontsize=12)
    setup_ax(ax2, bounds, crs)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "isopach_confidence.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'isopach_confidence.png'}")

    # ── Figure 3: Thickness histogram ─────────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    t_valid = thick[np.isfinite(thick)].ravel()
    ax3.hist(t_valid, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax3.axvline(np.median(t_valid), color="red", linewidth=1.5, linestyle="--",
                label=f"Median = {np.median(t_valid):.3f} m")
    ax3.set_xlabel("Sediment Thickness (m)", fontsize=11)
    ax3.set_ylabel("Pixel Count", fontsize=11)
    ax3.set_title("Isopach Distribution", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "isopach_histogram.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'isopach_histogram.png'}")

    plt.show()

    # stats
    print(f"\nThickness: {t_valid.min():.3f} ~ {t_valid.max():.3f} m, "
          f"median={np.median(t_valid):.3f} m, std={t_valid.std():.3f} m")
    conf_valid = conf[np.isfinite(conf) & (conf > 0)].ravel()
    if len(conf_valid):
        print(f"Confidence: near-measured (>=128): "
              f"{(conf_valid >= 128).sum()} ({100 * (conf_valid >= 128).sum() / len(conf_valid):.1f}%)")


if __name__ == "__main__":
    main()