import numpy as np
import matplotlib.pyplot as plt
import utils
import os
import glob
import pandas as pd
from matplotlib.ticker import FuncFormatter

# ================= 參數設定 =================
JSF_DIR = r'data\sbp'
MBES_FILE = r'data\multibeam\G1m_142m.txt'

# 繪圖參數
FIG_SIZE = (10, 9)
DPI = 200

# 顏色設定
DEPTH_CMAP = 'jet_r'  # 水深顏色
TRACK_COLOR = 'white'  # 測線顏色
TRACK_SIZE = 0.1  # 測線點大小 (越小越精細)
DEPTH_POINT_SIZE = 0.5  # 水深點大小 (因為點很多，設小一點才不會糊成一團)


# ===========================================

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


def get_sbp_tracks_only():
    jsf_files = sorted(glob.glob(os.path.join(JSF_DIR, "*.jsf")))
    lons, lats = [], []
    print(f"Loading SBP tracks from {len(jsf_files)} files...")
    for jsf in jsf_files:
        data = utils.read_jsf(jsf)
        for p in data:
            lons.append(p['lon'])
            lats.append(p['lat'])
    return np.array(lons), np.array(lats)


def main():
    if not os.path.exists(MBES_FILE):
        print("Error: Multibeam file not found.")
        return

    # 1. 讀取原始資料
    print(f"Loading Raw Multibeam data...")
    try:
        df = pd.read_csv(MBES_FILE, sep=r'\s+', header=None, names=['x', 'y', 'z'])
        # 座標轉換
        mb_lons, mb_lats = utils.twd97_to_wgs84(df['x'].values, df['y'].values)
        mb_depths = np.abs(df['z'].values)  # 取絕對值
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. 建立畫布
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.set_facecolor('white')

    # 3. 直接繪製水深點 (Raw Scatter Plot)
    print("Plotting Raw Depth Points...")
    # 使用 scatter 直接畫出每一個測點，不經過任何網格運算
    sc = ax.scatter(mb_lons, mb_lats, c=mb_depths, cmap=DEPTH_CMAP,
                    s=DEPTH_POINT_SIZE, alpha=1.0, linewidth=0, zorder=0)

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Depth (m)', rotation=270, labelpad=15)
    cbar.solids.set_alpha(1.0)

    # 4. 直接繪製測線軌跡
    sbp_lons, sbp_lats = get_sbp_tracks_only()
    if len(sbp_lons) > 0:
        print("Plotting Survey Tracks...")
        ax.scatter(sbp_lons, sbp_lats, c=TRACK_COLOR, s=TRACK_SIZE, zorder=1)

    # 5. 修飾圖表
    ax.set_title("Bathymetric chert & Tracks")

    # 格式化座標軸
    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    plt.setp(ax.get_xticklabels(), ha='right')

    # 設定範圍與置中
    min_x, max_x = np.min(mb_lons), np.max(mb_lons)
    min_y, max_y = np.min(mb_lats), np.max(mb_lats)

    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    dx, dy = max_x - min_x, max_y - min_y

    # 計算 10:9 比例
    fig_ratio = FIG_SIZE[0] / FIG_SIZE[1]
    if dx / dy > fig_ratio:
        target_dx = dx
        target_dy = dx / fig_ratio
    else:
        target_dy = dy
        target_dx = dy * fig_ratio

    target_dx *= 1.05
    target_dy *= 1.05

    ax.set_xlim(cx - target_dx / 2, cx + target_dx / 2)
    ax.set_ylim(cy - target_dy / 2, cy + target_dy / 2)
    ax.set_aspect('equal')

    ax.grid(True, linestyle='-', alpha=0.2)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()