import glob
import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation

from src import utils

# ================= 參數設定 =================
JSF_DIR = r"data\sbp"
MBES_FILE = r"data\multibeam\G1m_142m.txt"

FIG_SIZE = (10, 9)
DPI = 200
GRID_RES = 1000j

# 背景顏色 (Jet_r)
DEPTH_CMAP = "turbo_r"


# ============================================


def to_dms(x):
    degrees = int(x)
    minutes_float = (x - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    return degrees, minutes, seconds


def lon_formatter(x, pos):
    d, m, s = to_dms(x)
    return f"{d}°{m:02d}'{s:02.0f}\"E"


def lat_formatter(x, pos):
    d, m, s = to_dms(x)
    return f"{d}°{m:02d}'{s:02.0f}\"N"


def get_specific_tracks(n_tracks=3):
    all_jsf_files = sorted(glob.glob(os.path.join(JSF_DIR, "*.jsf")))
    if len(all_jsf_files) == 0:
        print("Error: No .jsf files found.")
        return []

    target_files = all_jsf_files[:n_tracks]
    tracks_data = []
    for i, jsf in enumerate(target_files):
        data = utils.read_jsf(jsf)
        t_lons, t_lats = [], []
        for p in data:
            t_lons.append(p["lon"])
            t_lats.append(p["lat"])
        if t_lons:
            tracks_data.append(
                {
                    "id": str(i + 1),
                    "filename": os.path.basename(jsf),
                    "lons": np.array(t_lons),
                    "lats": np.array(t_lats),
                }
            )
    return tracks_data


def main():
    if not os.path.exists(MBES_FILE):
        print("Error: MBES file not found.")
        return

    # 1. 讀取與處理地形
    df = pd.read_csv(MBES_FILE, sep=r"\s+", header=None, names=["x", "y", "z"])
    bg_lons, bg_lats = utils.twd97_to_wgs84(df["x"].values, df["y"].values)
    bg_depths = df["z"].values

    min_x, max_x = np.min(bg_lons), np.max(bg_lons)
    min_y, max_y = np.min(bg_lats), np.max(bg_lats)
    data_width = max_x - min_x
    data_height = max_y - min_y

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.set_facecolor("white")

    # 網格化
    padding_x = data_width * 0.05
    padding_y = data_height * 0.05
    num_grid = int(abs(GRID_RES))
    x_lin = np.linspace(min_x - padding_x, max_x + padding_x, num_grid)
    y_lin = np.linspace(min_y - padding_y, max_y + padding_y, num_grid)
    grid_x, grid_y = np.meshgrid(x_lin, y_lin)

    grid_depth = griddata(
        (bg_lons, bg_lats), bg_depths, (grid_x, grid_y), method="linear"
    )

    # 簡單遮罩
    idx_x = ((bg_lons - x_lin[0]) / (x_lin[-1] - x_lin[0]) * (num_grid - 1)).astype(int)
    idx_y = ((bg_lats - y_lin[0]) / (y_lin[-1] - y_lin[0]) * (num_grid - 1)).astype(int)
    shape_mask = np.zeros((num_grid, num_grid), dtype=bool)
    valid = (idx_x >= 0) & (idx_x < num_grid) & (idx_y >= 0) & (idx_y < num_grid)
    shape_mask[idx_y[valid], idx_x[valid]] = True
    shape_mask = binary_dilation(shape_mask, iterations=3)
    grid_depth[~shape_mask] = np.nan
    grid_depth = np.abs(grid_depth)

    # 2. 繪製背景 (zorder=0, alpha=0.8)
    pcm = ax.pcolormesh(
        grid_x, grid_y, grid_depth, cmap=DEPTH_CMAP, shading="auto", alpha=1, zorder=0
    )

    # cbar = plt.colorbar(pcm, ax=ax, fraction=0.03, pad=0.04)
    # cbar.set_label('Depth (m)', rotation=270, labelpad=15)

    # 3. 繪製軌跡 (zorder=1, Simple Style)
    tracks = get_specific_tracks(n_tracks=3)
    print("Plotting tracks...")

    for i, trk in enumerate(tracks):
        lons = trk["lons"]
        lats = trk["lats"]
        label_id = trk["id"]

        # A. 畫軌跡線 (使用 Matplotlib 預設顏色循環, 寬度 2px, zorder=1)
        # 不指定 color，讓它自動變換 (藍 -> 橘 -> 綠 ...)
        p = ax.plot(
            lons,
            lats,
            linewidth=2,
            label=label_id,
            zorder=1,
            path_effects=[pe.withStroke(linewidth=2, foreground="black")],
        )

        # 取得剛剛畫線用的顏色，讓起點顏色一致
        line_color = p[0].get_color()

        # B. 標示起點 (單純一個圈, 無文字)
        start_lon, start_lat = lons[0], lats[0]
        ax.scatter(start_lon, start_lat, color=line_color, s=50, marker="o", zorder=1)

    # 4. 修飾
    # ax.set_title(f"Survey Tracks")
    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    plt.setp(ax.get_xticklabels(), ha="right")

    # 簡單圖例
    ax.legend(title="Track ID", loc="upper right")

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_aspect("equal")

    # # 自動縮放 (Zoom)
    # all_lons = np.concatenate([t['lons'] for t in tracks])
    # all_lats = np.concatenate([t['lats'] for t in tracks])
    # mid_lon = (np.min(all_lons) + np.max(all_lons)) / 2
    # mid_lat = (np.min(all_lats) + np.max(all_lats)) / 2
    # span_lon = np.max(all_lons) - np.min(all_lons)
    # span_lat = np.max(all_lats) - np.min(all_lats)
    # zoom = 1.3
    # ax.set_xlim(mid_lon - span_lon * zoom / 2, mid_lon + span_lon * zoom / 2)
    # ax.set_ylim(mid_lat - span_lat * zoom / 2, mid_lat + span_lat * zoom / 2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
