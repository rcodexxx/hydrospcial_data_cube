# scripts/plot/plot_backscatter.py
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Transformer
from skimage import exposure

from src.config import ROOT

# ─── 定義檔案路徑 ───
MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"

# 輸入：一般聲納影像、已校正BS、K-means分類圖
IMG_TIF = ROOT / "outputs/tif/sss_imagery_hf.tif"
BS_TIF = ROOT / "outputs/tif/sss_backscatter_hf.tif"
LBL_TIF = ROOT / "outputs/tif/sss_clusters_hf.tif"  # 新增 Cluster 輸入

# 輸出：三張不同的圖檔
OUT_IMG_FIG = ROOT / "outputs/figures/sss_imagery_hf.png"
OUT_BS_FIG = ROOT / "outputs/figures/sss_backscatter_hf.png"
OUT_LBL_FIG = ROOT / "outputs/figures/sss_clusters_hf.png"  # 新增 Cluster 輸出

CUT_NORTH = 2449050  # remove turning artifact
N_CLUSTERS = 7  # 設定有幾個群


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

    # 根據不同圖層類型，設定不同的渲染邏輯
    if plot_type == "imagery":
        vmin, vmax = np.percentile(data[valid_mask], [1, 99])
        norm_data = np.clip(data, vmin, vmax)
        norm_data = (norm_data - vmin) / (vmax - vmin)
        norm_data[~valid_mask] = 0.0
        enhanced_data = exposure.equalize_adapthist(norm_data, clip_limit=0.03)
        enhanced_data[~valid_mask] = np.nan
        plot_data = enhanced_data
        cmap = "copper"
        plot_vmin, plot_vmax = 0, 1
        cbar_label = "Relative Intensity (Enhanced)"

    elif plot_type == "cluster":
        # 畫分類圖：使用類別型色碼 tab10
        plot_data = data
        cmap = plt.get_cmap("tab10", N_CLUSTERS)
        plot_vmin, plot_vmax = -0.5, N_CLUSTERS - 0.5  # 讓色階完美對齊整數
        cbar_label = "Acoustic Facies (Cluster ID)"

    else:  # plot_type == "bs"
        vmin, vmax = np.percentile(data[valid_mask], [2, 98])
        plot_data = data
        cmap = "gray_r"
        plot_vmin, plot_vmax = vmin, vmax
        cbar_label = "Backscatter Strength (dB)"

    im = ax.imshow(
        plot_data,
        cmap=cmap,
        origin="upper",
        aspect="equal",
        extent=extent,
        vmin=plot_vmin,
        vmax=plot_vmax,
    )

    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.05)

    # 分類圖的 Colorbar 需要特別處理，讓刻度只顯示整數
    if plot_type == "cluster":
        cbar = plt.colorbar(im, cax=cax, label=cbar_label, ticks=range(N_CLUSTERS))
    else:
        cbar = plt.colorbar(im, cax=cax, label=cbar_label)

    ax.set_title(title, fontsize=14, fontweight="bold")
    setup_ax(ax, bounds, tr)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    dem, bounds, crs = load_tif(MBES_TIF)
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    rows = dem.shape[0]
    res = (bounds.top - bounds.bottom) / rows
    cut_row = int((bounds.top - CUT_NORTH) / res)
    mask = ~np.isnan(dem)
    mask[:cut_row, :] = False

    # 1. 輸出影像圖 (Imagery)
    img_data, _, _ = load_tif(IMG_TIF, nodata=-9999.0)
    img_data = np.where(mask, img_data, np.nan)
    render_plot(
        img_data,
        bounds,
        tr,
        OUT_IMG_FIG,
        "Mudan Reservoir - SSS Imagery",
        plot_type="imagery",
    )

    # 2. 輸出背向散射數值圖 (BS)
    bs_data, _, _ = load_tif(BS_TIF, nodata=-9999.0)
    bs_data = np.where(mask, bs_data, np.nan)
    render_plot(
        bs_data,
        bounds,
        tr,
        OUT_BS_FIG,
        "Mudan Reservoir - SSS Backscatter",
        plot_type="bs",
    )

    # 3. 輸出 K-means 分類分佈圖 (Clusters)
    # 注意：Cluster 的 nodata 是 255
    lbl_data, _, _ = load_tif(LBL_TIF, nodata=255)
    lbl_data = np.where(mask, lbl_data, np.nan)
    render_plot(
        lbl_data,
        bounds,
        tr,
        OUT_LBL_FIG,
        "Mudan Reservoir - Acoustic Facies (K-means)",
        plot_type="cluster",
    )


if __name__ == "__main__":
    main()
