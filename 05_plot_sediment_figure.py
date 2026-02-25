import glob
import os

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from pykrige.ok import OrdinaryKriging
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation

from src import utils

# ================= 參數設定 (完全保留您的設定) =================
JSF_DIR = r"data\sbp"
MBES_FILE = r"data\multibeam\G1m_142m.txt"

CC_VALUE = 8.5867e07
BLANKING = 1

# 繪圖參數
FIG_SIZE = (10, 9)
DPI = 200
GRID_RES = 1000j
SMOOTH_SIGMA = 4

# RL 顏色
RL_VMIN = 0
RL_VMAX = 35
RL_CMAP = "jet_r"
RL_ALPHA = 0.8

# 等深線參數
CONTOUR_THIN_STEP = 7 / 4
CONTOUR_THICK_STEP = 7
LABEL_INTERVAL = 7


# ========================================================


def to_dms(x):
    """計算度分秒數值"""
    degrees = int(x)
    minutes_float = (x - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    return degrees, minutes, seconds


def lon_formatter(x, pos):
    """經度格式化 (加 E)"""
    d, m, s = to_dms(x)
    return f"{d}°{m:02d}'{s:02.0f}\"E"


def lat_formatter(x, pos):
    """緯度格式化 (加 N)"""
    d, m, s = to_dms(x)
    return f"{d}°{m:02d}'{s:02.0f}\"N"


def get_sbp_data_points():
    jsf_files = sorted(glob.glob(os.path.join(JSF_DIR, "*.jsf")))
    track_lons, track_lats, track_rls = [], [], []

    print(f"Loading SBP data from {len(jsf_files)} files...")
    for jsf in jsf_files:
        data = utils.read_jsf(jsf)
        for p in data:
            amps = p["amps"]
            if len(amps) < 100:
                continue
            search_region = amps[BLANKING:]
            if len(search_region) == 0:
                continue
            idx_max = np.argmax(search_region) + BLANKING
            amp_1 = float(amps[idx_max])
            r_1 = float(idx_max)
            if amp_1 > 0:
                R = (r_1 * amp_1) / CC_VALUE
                if R > 1.0:
                    R = 1.0
                if R < 0.0001:
                    R = 0.0001
                RL = -20 * np.log10(R)
                track_lons.append(p["lon"])
                track_lats.append(p["lat"])
                track_rls.append(RL)
    return np.array(track_lons), np.array(track_lats), np.array(track_rls)


def main():
    print("Initializing Map...")

    if not os.path.exists(MBES_FILE):
        print("Error: Multibeam file not found.")
        return

    print(f"Loading Multibeam data: {os.path.basename(MBES_FILE)}...")
    try:
        df = pd.read_csv(MBES_FILE, sep=r"\s+", header=None, names=["x", "y", "z"])
        bg_lons, bg_lats = utils.twd97_to_wgs84(df["x"].values, df["y"].values)
        bg_depths = df["z"].values

        # 取得資料範圍
        min_x, max_x = np.min(bg_lons), np.max(bg_lons)
        min_y, max_y = np.min(bg_lats), np.max(bg_lats)
        data_width = max_x - min_x
        data_height = max_y - min_y

    except Exception as e:
        print(f"Error: {e}")
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.set_facecolor("white")

    # 2. 建立網格
    padding_x = data_width * 0.01
    padding_y = data_height * 0.01
    num_grid = int(abs(GRID_RES))
    x_lin = np.linspace(min_x - padding_x, max_x + padding_x, num_grid)
    y_lin = np.linspace(min_y - padding_y, max_y + padding_y, num_grid)
    grid_x, grid_y = np.meshgrid(x_lin, y_lin)

    # 3. 製作遮罩
    idx_x = (
        (bg_lons - (min_x - padding_x))
        / ((max_x + padding_x) - (min_x - padding_x))
        * (num_grid - 1)
    ).astype(int)
    idx_y = (
        (bg_lats - (min_y - padding_y))
        / ((max_y + padding_y) - (min_y - padding_y))
        * (num_grid - 1)
    ).astype(int)
    valid_mask = (idx_x >= 0) & (idx_x < num_grid) & (idx_y >= 0) & (idx_y < num_grid)
    shape_mask = np.zeros((num_grid, num_grid), dtype=bool)
    shape_mask[idx_y[valid_mask], idx_x[valid_mask]] = True
    shape_mask = binary_dilation(shape_mask, iterations=3)

    # 4. 處理地形資料
    grid_depth = griddata(
        (bg_lons, bg_lats), bg_depths, (grid_x, grid_y), method="linear"
    )
    grid_depth[~shape_mask] = np.nan
    grid_depth = np.abs(grid_depth)  # 取絕對值

    # 5. 繪製等深線
    print("Plotting Contours...")
    z_min, z_max = np.nanmin(grid_depth), np.nanmax(grid_depth)
    base_level = np.floor(z_min)
    top_level = np.ceil(z_max)

    # # 細線 (實線)
    # levels_thin = np.arange(base_level, top_level + 1, CONTOUR_THIN_STEP)
    # ax.contour(grid_x, grid_y, grid_depth, levels=levels_thin,
    #            colors='black', linewidths=0.3, alpha=0.8, zorder=2, linestyles='solid')

    # 粗線 (實線)
    levels_thick = np.arange(
        np.round(base_level / 7) * 7, top_level + 1, CONTOUR_THICK_STEP
    )
    CS_thick = ax.contour(
        grid_x,
        grid_y,
        grid_depth,
        levels=levels_thick,
        colors="black",
        linewidths=0.6,
        alpha=1.0,
        zorder=2,
        linestyles="solid",
    )

    # # 標籤過濾 (避免數字太多)
    # levels_label = [l for l in levels_thick if l % LABEL_INTERVAL == 0]
    # clabels = ax.clabel(CS_thick, levels=levels_label, inline=True, fontsize=8, fmt='%1.0f', colors='black')

    # # 簡單的距離過濾算法
    # min_dist = (max_x - min_x) * 0.15  # 設定過濾半徑
    # kept_labels = []
    # for label in clabels:
    #     text = label.get_text()
    #     x, y = label.get_position()
    #     too_close = False
    #     for (k_text, kx, ky) in kept_labels:
    #         if text == k_text:
    #             if np.sqrt((x - kx) ** 2 + (y - ky) ** 2) < min_dist:
    #                 too_close = True
    #                 break
    #     if too_close:
    #         label.set_visible(False)
    #     else:
    #         kept_labels.append((text, x, y))

    # 6. 處理 RL 資料
    sbp_lons, sbp_lats, sbp_rls = get_sbp_data_points()
    if len(sbp_rls) > 0:
        print("Processing RL Layer with Kriging...")

        # --- [新增] 克里金插值模型建立 ---
        # 建立普通克里金模型 (Ordinary Kriging)
        # variogram_model 可以選擇 'spherical', 'exponential', 'gaussian', 'linear' 等
        OK = OrdinaryKriging(
            sbp_lons,
            sbp_lats,
            sbp_rls,
            variogram_model="exponential",
            nlags=15,  # 變異函數的計算階層數
            enable_plotting=False,  # 設為 True 可以查看半變異函數圖
        )

        # --- [新增] 執行插值 ---
        # PyKrige 的 'grid' 模式直接吃一維的 X 和 Y 軸座標 (就是你前面建好的 x_lin, y_lin)
        # 這裡會花費較多時間計算
        print("Executing Kriging interpolation (This might take a while...)")
        grid_rl_krige, ss_krige = OK.execute(
            "grid", x_lin, y_lin, backend="loop", n_closest_points=100
        )

        # 提取運算結果 (轉為 numpy array)
        grid_rl = np.array(grid_rl_krige, dtype=float)

        # 套用你原本寫好的遮罩 (將地形範圍外的資料設為 NaN)
        grid_rl[~shape_mask] = np.nan

        # (由於克里金本身就會根據空間結構進行平滑，通常不需要再加 gaussian_filter)
        # 如果你覺得結果還是有點破碎，可以再把這行加回來：
        # grid_rl = gaussian_filter(grid_rl, sigma=SMOOTH_SIGMA)

        # 1. 計算資料的 min 和 max
        data_vmin = np.nanmin(grid_rl)
        data_vmax = np.nanmax(grid_rl)

        print(f"RL Data Range: Min={data_vmin:.2f}, Max={data_vmax:.2f}")

        # 2. 繪圖時使用動態範圍
        mesh = ax.pcolormesh(
            grid_x,
            grid_y,
            grid_rl,
            cmap=RL_CMAP,
            vmin=data_vmin,
            vmax=data_vmax,
            shading="auto",
            alpha=RL_ALPHA,
            zorder=1,
        )

        # 3. Colorbar 也使用動態範圍
        norm = mcolors.Normalize(vmin=data_vmin, vmax=data_vmax)

        sm = cm.ScalarMappable(norm=norm, cmap=RL_CMAP)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("RL (dB)", rotation=270, labelpad=15)
        cbar.solids.set_alpha(1.0)
    else:
        print("No SBP data found.")

    # 7. 最終修飾
    ax.set_title(f"RL Distribution")

    # 座標格式設定 (E/N)
    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    plt.setp(ax.get_xticklabels(), ha="right")

    ax.grid(True, linestyle="-", alpha=0.2, zorder=0)

    # ----------------------------------------------------
    # 自動置中算法 (解決 Ignoring fixed y limits 警告)
    # ----------------------------------------------------
    rows = np.any(shape_mask, axis=1)
    cols = np.any(shape_mask, axis=0)
    if np.any(rows) and np.any(cols):
        ymin, ymax = y_lin[np.where(rows)[0][[0, -1]]]
        xmin, xmax = x_lin[np.where(cols)[0][[0, -1]]]

        # 資料中心與尺寸
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        dx, dy = xmax - xmin, ymax - ymin

        # 畫布比例 (10/9)
        fig_ratio = FIG_SIZE[0] / FIG_SIZE[1]

        # 根據比例決定擴充寬或高
        if dx / dy > fig_ratio:
            target_dx = dx
            target_dy = dx / fig_ratio
        else:
            target_dy = dy
            target_dx = dy * fig_ratio

        # 加上 5% 緩衝
        target_dx *= 1.05
        target_dy *= 1.05

        ax.set_xlim(cx - target_dx / 2, cx + target_dx / 2)
        ax.set_ylim(cy - target_dy / 2, cy + target_dy / 2)

    ax.set_aspect("equal")  # 鎖定比例

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
