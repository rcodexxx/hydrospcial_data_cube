# 02_plot_segment.py
import glob
import os

import matplotlib.pyplot as plt

from src import utils

# --- 設定區 (Config) ---
XYZ_PATH = r"data\multibeam\G1m_142m.txt"
JSF_DIR = r"data\sbp"

TARGET_IDX = 5
START_M = 630.0  # 開始距離 (m)
END_M = 780.0  # 結束距離 (m)


def main():
    # 1. 讀底圖
    df = utils.read_xyz(XYZ_PATH)
    bg_lon, bg_lat = utils.twd97_to_wgs84(df["x"].values, df["y"].values)

    # 2. 讀特定 JSF
    jsf_files = sorted(glob.glob(os.path.join(JSF_DIR, "*.jsf")))
    target_file = jsf_files[TARGET_IDX]
    print(f"Analyzing: {os.path.basename(target_file)}")

    lons, lats, pings = utils.read_jsf_track(target_file)

    # 3. 計算截取區間
    dists = utils.calc_cumulative_dist(lons, lats)
    mask = (dists >= START_M) & (dists <= END_M)

    # 取得截取資料
    seg_lons = lons[mask]
    seg_lats = lats[mask]
    seg_pings = pings[mask]

    print(f"Segment Pings: #{seg_pings[0]} - #{seg_pings[-1]}")

    # 4. 繪圖
    plt.figure(figsize=(10, 8), dpi=120)

    # 底圖
    plt.scatter(bg_lon, bg_lat, c=df["z"], s=0.01, cmap="viridis", alpha=0.5)

    # 完整路徑 (灰)
    plt.plot(lons, lats, c="gray", lw=1, alpha=0.5, label="Full Track")

    # 截取路徑 (紅)
    plt.plot(seg_lons, seg_lats, c="red", lw=3, label="Target Segment")

    # # 標示起終點
    # plt.text(seg_lons[0], seg_lats[0], f"{int(START_M)}m", color='white', fontweight='bold', bbox=dict(fc='red'))
    # plt.text(seg_lons[-1], seg_lats[-1], f"{int(END_M)}m", color='white', fontweight='bold', bbox=dict(fc='red'))

    plt.title(f"Segment Analysis")
    plt.axis("equal")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
