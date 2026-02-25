import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# ================= 參數設定 =================
# 1. 物理範圍 (配合地圖 0~25)
PHYSICAL_VMIN = 0
PHYSICAL_VMAX = 25

# 2. 顯示範圍 (0~25)
VIEW_VMIN = 0.0
VIEW_VMAX = 25.0

# 3. 底質分類
SEDIMENT_DATA = [
    (0.0, 8.2, "Coarse Sand"),
    (8.2, 8.8, "Fine Sand"),
    (8.8, 9.5, "Very Fine Sand"),
    (9.5, 11.0, "Silty Sand"),
    (11.0, 12.8, "Sand-Silt-Clay"),
    (12.8, 14.3, "Sandy Silt"),
    (14.3, 15.6, "Clayey Silt"),
    (15.6, 25.0, "Silty Clay"),
]

CMAP_NAME = "jet_r"
FIG_SIZE = (6, 8)
DPI = 200


# ===========================================


def main():
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    # 【修改 1：讓 Colorbar 變瘦】
    # 原本是 left=0.45, right=0.85 (寬度 0.4)
    # 現在改 left=0.55, right=0.70 (寬度 0.15) -> 變得很瘦
    fig.subplots_adjust(left=0.55, right=0.70, top=0.95, bottom=0.05)

    # 1. 繪製漸層
    gradient = np.linspace(PHYSICAL_VMIN, PHYSICAL_VMAX, 256).reshape(-1, 1)
    ax.imshow(
        gradient,
        aspect="auto",
        cmap=CMAP_NAME,
        origin="lower",
        extent=[0, 1, PHYSICAL_VMIN, PHYSICAL_VMAX],
    )

    ax.set_ylim(VIEW_VMIN, VIEW_VMAX)
    ax.set_xlim(0, 1)
    ax.set_xticks([])

    # Y軸設定
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(labelsize=11)
    ax.set_ylabel("RL (dB)", fontsize=12, labelpad=15)
    ax.minorticks_on()

    # 2. 繪製左側標籤與線條
    print("Plotting thin legend...")

    last_text_y = -999

    for low, high, label in SEDIMENT_DATA:
        mid = (low + high) / 2

        if low > VIEW_VMAX:
            continue
        if high < VIEW_VMIN:
            continue

        # --- 防重疊邏輯 ---
        text_y = mid
        if mid < VIEW_VMIN:
            text_y = VIEW_VMIN + 0.5
        if mid > VIEW_VMAX:
            text_y = VIEW_VMAX - 0.5

        min_dist = 0.8
        if text_y - last_text_y < min_dist:
            text_y = last_text_y + min_dist

        last_text_y = text_y

        # 【修改 2：線條改成黑色】
        # color='black'
        # alpha=0.3 (設透明一點，才不會把顏色蓋得太醜，您可以依喜好改成 0.5 或 1.0)
        ax.axhline(low, color="black", linewidth=0.8, alpha=0.4)
        if high <= VIEW_VMAX:
            ax.axhline(high, color="black", linewidth=0.8, alpha=0.4)

        # 畫引導線
        ax.plot([0, -0.2], [mid, text_y], color="black", linewidth=0.8, clip_on=False)

        # 寫文字 (稍微往左移一點，因為 Bar 變瘦了)
        height = high - low
        font_size = 11 if height > 1.5 else 10

        ax.text(
            -0.25,
            text_y,
            label,
            ha="right",
            va="center",
            fontsize=font_size,
            color="black",
        )

    # ax.set_title("Sediment Classification", fontsize=14, pad=15)

    plt.show()


if __name__ == "__main__":
    main()
