import glob
import os
import struct
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from src import calculation, config, data_loader, utils

plt.rcParams["contour.negative_linestyle"] = "solid"


def main():
    print("Reading Multibeam background...")
    try:
        mb_df = data_loader.read_xyz(config.MB_PATH)
        mb_xy = mb_df[["x", "y"]].values
        mb_z = mb_df["z"].values
    except FileNotFoundError:
        print(f"Error: Multibeam file not found at {config.MB_PATH}")
        return
    except Exception as e:
        print(f"Error: Failed to read Multibeam file. {e}")
        return

    jsf_files = sorted(config.SBP_PATH.glob("*.jsf"))
    if not jsf_files:
        print(f"Error: No JSF file found at {config.SBP_PATH}")
        return

    print(f"Found {len(jsf_files)} JSF file(s). Calculating RL...")

    # 收集所有檔案的有效 RL 資料
    all_lons, all_lats, all_rls = [], [], []

    for jsf in jsf_files:
        data = data_loader.read_jsf(jsf)
        if not data:
            continue

        for p in data:
            rl = calculation.calculate_ping_rl(p["amps"], config.CC)
            if rl is not None:
                all_lons.append(p["lon"])
                all_lats.append(p["lat"])
                all_rls.append(rl)

    lons_arr = np.array(all_lons)
    lats_arr = np.array(all_lats)
    rls_arr = np.array(all_rls)

    if len(rls_arr) == 0:
        print("Error: No valid RL data calculated.")
        return

    print(f"Total valid pings: {len(rls_arr)}. Converting WGS84 to TWD97...")
    # 將 SBP 經緯度轉為 TWD97 進行精確空間運算
    sbp_x, sbp_y = utils.wgs84_to_twd97(lons_arr, lats_arr)
    sbp_xy = np.column_stack((sbp_x, sbp_y))

    MAX_KRIGING_POINTS = 1800
    if len(sbp_xy) > MAX_KRIGING_POINTS:
        print(
            f"Data points ({len(sbp_xy)}) exceed Kriging limit. Downsampling to {MAX_KRIGING_POINTS}..."
        )
        # 每隔 N 個點取樣一次，保持空間均勻性
        step = len(sbp_xy) // MAX_KRIGING_POINTS
        kriging_xy = sbp_xy[::step]
        kriging_rls = rls_arr[::step]
    else:
        kriging_xy = sbp_xy
        kriging_rls = rls_arr

    print(f"Creating grid (Resolution: {1}m)...")
    x_min, x_max = mb_xy[:, 0].min(), mb_xy[:, 0].max()
    y_min, y_max = mb_xy[:, 1].min(), mb_xy[:, 1].max()

    grid_x, grid_y = np.mgrid[x_min:x_max:1, y_min:y_max:1]

    # ==========================================
    # ★ 替換：使用 Ordinary Kriging 進行插值
    # ==========================================
    from pykrige.ok import OrdinaryKriging

    print("Initializing Ordinary Kriging... (this may take a moment)")
    # variogram_model 可選: 'linear', 'power', 'spherical', 'gaussian', 'exponential'
    # 'spherical' (球狀模型) 在地質與聲納資料中最為常用
    OK = OrdinaryKriging(
        kriging_xy[:, 0],
        kriging_xy[:, 1],
        kriging_rls,
        variogram_model="spherical",
        verbose=False,
        enable_plotting=False,
    )

    print("Executing Kriging interpolation...")
    # 執行插值。將 2D 網格展平 (ravel) 以符合 pykrige 格式，計算後再重塑回 2D
    z_krig, ss_krig = OK.execute("points", grid_x.ravel(), grid_y.ravel())
    grid_rl = z_krig.reshape(grid_x.shape)

    print(f"Applying KDTree distance mask (Threshold: {5}m)...")
    tree = cKDTree(mb_xy)
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    distances, _ = tree.query(grid_points)
    distances = distances.reshape(grid_x.shape)

    # 將距離過遠的網格挖空設為 NaN
    grid_rl[distances > 5] = np.nan

    print("Gridding Multibeam depth for contours...")
    # 若 MB 資料量過大 (超過 5 萬點)，進行適度降採樣以加速插值
    step_mb = max(1, len(mb_xy) // 50000)

    # 這裡用 linear 插值就很平滑了
    grid_z = griddata(
        mb_xy[::step_mb], mb_z[::step_mb], (grid_x, grid_y), method="linear"
    )
    # 套用跟 RL 一模一樣的邊界遮罩，確保等深線不會畫到陸地上
    grid_z[distances > 5] = np.nan

    print("Converting grid back to WGS84 for plotting...")
    # 轉回經緯度以利繪製符合 DMS 習慣的地圖
    grid_lon, grid_lat = utils.twd97_to_wgs84(grid_x, grid_y)

    print("Plotting map...")
    fig, ax = plt.subplots(**config.FIGURE_STYLE)

    # 繪製 RL 熱力圖
    pcm = ax.pcolormesh(
        grid_lon,
        grid_lat,
        grid_rl,
        cmap="turbo_r",
        shading="nearest",
        rasterized=True,
        vmin=0,
        vmax=25,
    )

    # # 疊加測線軌跡 (降低粗度與透明度避免喧賓奪主)
    # for jsf in jsf_files:
    #     data = data_loader.read_jsf(jsf)
    #     if not data:
    #         continue
    #     lons = [p["lon"] for p in data]
    #     lats = [p["lat"] for p in data]
    #     ax.plot(lons, lats, color="w", linewidth=0.5, alpha=0.4)

    # 找出深度的最大與最小值，並取整數
    z_min, z_max = np.nanmin(grid_z), np.nanmax(grid_z)
    min_lvl = int(np.floor(z_min))
    max_lvl = int(np.ceil(z_max))

    # 建立 1m 間距的所有整數陣列 (細線)
    levels_minor = np.arange(min_lvl, max_lvl + 1, 1)
    # 從中篩選出 7 的倍數 (粗線)
    levels_major = [lvl for lvl in levels_minor if lvl % 7 == 0]

    # 畫細線
    ax.contour(
        grid_lon,
        grid_lat,
        grid_z,
        levels=levels_minor,
        colors="black",
        linewidths=0.3,
        alpha=0.5,
    )

    # 畫粗線 (lw=1.0)
    cs = ax.contour(
        grid_lon,
        grid_lat,
        grid_z,
        levels=levels_major,
        colors="black",
        linewidths=0.6,
        alpha=0.7,
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt=lambda x: f"{abs(x):.0f}")
    cax = fig.add_axes([0.613, 0.185, 0.25, 0.03])

    cbar = fig.colorbar(
        pcm,
        cax=cax,
        orientation="horizontal",  # 設置為水平
        ticks=[0, 5, 10, 15, 20, 25],  # 指定 0~30，每 5 一個 tick
    )
    cbar.set_label("RL (dB)", fontsize=10)

    utils.apply_dms_ticks(ax, step_seconds=10.0)
    utils.set_map_aspect(ax, lats=grid_lat.ravel())

    plt.show()


if __name__ == "__main__":
    main()
