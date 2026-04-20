# scripts/plot/plot_backscatter.py
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Transformer
from scipy.ndimage import median_filter

from src.config import ROOT

# ─── 定義檔案路徑 ───
MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
BS_TIF = ROOT / "outputs/tif/sss_backscatter_lf.tif"
LBL_TIF = ROOT / "outputs/tif/sss_clusters_lf.tif" 

OUT_BS_FIG = ROOT / "outputs/figures/sss_backscatter_lf.png"
OUT_LBL_FIG = ROOT / "outputs/figures/sss_clusters_lf.png" 

N_CLUSTERS = 5


def load_tif(path, nodata=None):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nd = src.nodata if nodata is None else nodata
        if nd is not None:
            data[data == nd] = np.nan
        return data, src.bounds, src.crs


def setup_ax(ax, bounds, tr):
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
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, color="white", linewidth=0.4, alpha=0.5, linestyle="--")


def render_plot(data, bounds, tr, out_path, title, plot_type="bs"):
    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    valid_mask = ~np.isnan(data)
    if not valid_mask.any():
        print(f"Warning: No valid data to plot for {title}")
        return

    if plot_type == "cluster":
        plot_data = data
        cmap = plt.get_cmap("tab10", N_CLUSTERS)
        plot_vmin, plot_vmax = -0.5, N_CLUSTERS - 0.5
        cbar_label = "Acoustic Facies (Cluster ID)"

    else:
        # 畫背向散射圖：擷取 2% ~ 98% 避免極端雜訊影響對比度
        vmin, vmax = np.percentile(data[valid_mask], [2, 98])
        plot_data = data
        cmap = "copper"
        plot_vmin, plot_vmax = vmin, vmax
        cbar_label = "Absolute Backscatter Strength (dB)"

    im = ax.imshow(
        plot_data,
        cmap=cmap,
        origin="upper",
        aspect="equal",
        extent=extent,
        vmin=plot_vmin,
        vmax=plot_vmax,
        interpolation="none" # 確保分類圖邊緣銳利
    )

    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.05)

    if plot_type == "cluster":
        cbar = plt.colorbar(im, cax=cax, label=cbar_label, ticks=range(N_CLUSTERS))
    else:
        cbar = plt.colorbar(im, cax=cax, label=cbar_label)

    ax.set_title(title, fontsize=14, fontweight="bold")
    setup_ax(ax, bounds, tr)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Successfully rendered and saved: {out_path.name}")


def main():
    dem, bounds, crs = load_tif(MBES_TIF)
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # 1. 輸出物理定標後的背向散射圖 (BS)
    bs_data, _, _ = load_tif(BS_TIF, nodata=-9999.0)
    
    # 🌟 記住 SSS 本身的真實範圍
    valid_bs_mask = ~np.isnan(bs_data) 
    
    bs_filled = np.nan_to_num(bs_data, nan=np.nanmean(bs_data[valid_bs_mask]))
    bs_smoothed = median_filter(bs_filled, size=3) 
    
    # 🌟 濾波完後，只保留 SSS 真正有掃到的地方
    bs_data = np.where(valid_bs_mask, bs_smoothed, np.nan) 
    
    render_plot(
        bs_data,
        bounds,
        tr,
        OUT_BS_FIG,
        "Mudan Reservoir - Calibrated SSS Backscatter (HF)",
        plot_type="bs",
    )

    # 2. 輸出 K-means 分類分佈圖 (Clusters)
    lbl_data, _, _ = load_tif(LBL_TIF, nodata=255)
    
    # 🌟 記住 SSS 分類圖的真實範圍
    valid_lbl_mask = ~np.isnan(lbl_data)
    
    lbl_filled = np.nan_to_num(lbl_data, nan=255)
    lbl_smoothed = median_filter(lbl_filled, size=7) 
    
    # 🌟 濾波完後，只保留 SSS 真正有掃到的地方
    lbl_data = np.where(valid_lbl_mask, lbl_smoothed, np.nan)
    
    render_plot(
        lbl_data,
        bounds,
        tr,
        OUT_LBL_FIG,
        f"Mudan Reservoir - Acoustic Facies (HF, K={N_CLUSTERS})",
        plot_type="cluster",
    )

if __name__ == "__main__":
    main()