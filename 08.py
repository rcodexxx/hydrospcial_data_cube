import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from src import utils

# ================= 參數設定 =================
JSF_DIR = r"data\sbp"
CC_VALUE = 8.5867e07
BLANKING = 10

# 繪圖設定
FIG_SIZE = (12, 10)
DPI = 150

# SBP 影像強度範圍 (太黑/太淡請調這裡)
SBP_VMIN = 0
SBP_VMAX = 20000

# RL 數值範圍
RL_Y_MIN = 0
RL_Y_MAX = 27


# ===========================================


def main():
    jsf_files = sorted(glob.glob(os.path.join(JSF_DIR, "*.jsf")))
    if not jsf_files:
        print("Error: No .jsf files found.")
        return

    target_file = jsf_files[2]
    filename = os.path.basename(target_file)
    print(f"Processing (Raw Mode): {filename} ...")

    raw_data = utils.read_jsf(target_file)

    matrix_amps = []
    list_rls = []
    list_pings = []
    max_samples = 0

    print("Reading raw samples...")
    for i, p in enumerate(raw_data):
        amps = np.array(p["amps"], dtype=float)

        # 紀錄最大的樣本數長度 (作為 Y 軸最大值)
        if len(amps) > max_samples:
            max_samples = len(amps)

        matrix_amps.append(amps)

        # --- 計算 RL (邏輯不變) ---
        if len(amps) < BLANKING:
            list_rls.append(np.nan)
        else:
            search_region = amps[BLANKING:]
            if len(search_region) == 0:
                list_rls.append(np.nan)
            else:
                idx_local = np.argmax(search_region)
                idx_global = idx_local + BLANKING
                amp_1 = float(amps[idx_global])
                r_1 = float(idx_global)

                if amp_1 > 0:
                    R = (r_1 * amp_1) / CC_VALUE
                    if R > 1.0:
                        R = 1.0
                    if R < 0.0001:
                        R = 0.0001
                    rl_val = -20 * np.log10(R)
                    list_rls.append(rl_val)
                else:
                    list_rls.append(np.nan)
        list_pings.append(i)

    # --- 建立影像矩陣 ---
    n_pings = len(matrix_amps)
    # 建立一個 [最大樣本數 x Ping數] 的矩陣
    sbp_image = np.zeros((max_samples, n_pings))

    for i, amps in enumerate(matrix_amps):
        length = len(amps)
        # 填入原始振幅值
        sbp_image[:length, i] = amps

    rl_curve = np.array(list_rls)
    pings = np.array(list_pings)

    print(f"Data Shape: {n_pings} pings x {max_samples} samples")

    # ================= 繪圖設定 =================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIG_SIZE, dpi=DPI, sharex=True)

    # --- 上圖：SBP 原始影像 ---
    # extent = [x_min, x_max, y_max, y_min]
    # Y軸：0 在最上面，max_samples 在最下面
    extent = [0, n_pings, max_samples, 0]

    ax1.imshow(
        sbp_image,
        aspect="auto",
        cmap="gray_r",
        vmin=SBP_VMIN,
        vmax=SBP_VMAX,
        interpolation="nearest",
        extent=extent,
    )

    ax1.set_ylabel("Sample Index", fontsize=12)  # Y軸改為樣本點編號
    ax1.set_title(f"3", fontsize=12, fontweight="bold")

    # --- 下圖：RL 曲線 ---
    ax2.plot(pings, rl_curve, color="#0055aa", linewidth=1)

    ax2.set_ylim(RL_Y_MIN, RL_Y_MAX)
    ax2.set_ylabel("Reflection Loss (dB)", fontsize=12)
    ax2.set_xlabel("Ping Number", fontsize=12)

    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
