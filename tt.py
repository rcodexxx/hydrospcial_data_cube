# scripts/build/prove_fluid_mud.py
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- 路徑設定 (請確認與你的專案路徑相符) ---
ROOT = Path(__file__)
RL_TIF = ROOT / "../outputs/tif/sbp_rl.tif"
BATHY_TIF = ROOT / "../outputs/tif/mbes_bathymetry.tif"
OUT_PLOT = ROOT / "../outputs/plots/fluid_mud_evidence.png"

# 你的 Hamilton 理論極限 (Silty clay / Clayey silt 的交界帶)
HAMILTON_LIMIT = 13.37


def main():
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

    print("讀取 RL 與 Bathymetry 資料中...")
    with rasterio.open(RL_TIF) as src_rl:
        rl_data = src_rl.read(1)
        rl_nodata = src_rl.nodata

    with rasterio.open(BATHY_TIF) as src_bathy:
        bathy_data = src_bathy.read(1)
        bathy_nodata = src_bathy.nodata

    # 展平陣列並過濾出兩個 TIF 都有有效數值的點
    valid_mask = (rl_data != rl_nodata) & np.isfinite(rl_data) & \
                 (bathy_data != bathy_nodata) & np.isfinite(bathy_data)

    rl_valid = rl_data[valid_mask]
    bathy_valid = bathy_data[valid_mask]

    print(f"有效數據點數量: {len(rl_valid)}")
    print(f"RL 範圍: {rl_valid.min():.2f} ~ {rl_valid.max():.2f} dB")

    # 計算有多少比例的資料超出了 Hamilton 極限
    exceed_mask = rl_valid > HAMILTON_LIMIT
    exceed_ratio = (exceed_mask.sum() / len(rl_valid)) * 100
    print(f"超出 Hamilton 極限 ({HAMILTON_LIMIT} dB) 的數據比例: {exceed_ratio:.1f}%")

    # --- 開始繪製證據圖 ---
    print("繪製流體泥分析圖表...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # [證據 1] RL 統計直方圖 (Histogram)
    ax1.hist(rl_valid, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(HAMILTON_LIMIT, color='red', linestyle='--', linewidth=2,
                label=f'Hamilton Limit (~{HAMILTON_LIMIT} dB)')
    ax1.set_title("Evidence 1: RL Statistical Distribution", fontweight='bold')
    ax1.set_xlabel("Reflection Loss (dB)")
    ax1.set_ylabel("Pixel Count")

    # 將大於極限的部分塗成深色，凸顯異常群集
    ax1.axvspan(HAMILTON_LIMIT, rl_valid.max(), color='red', alpha=0.1, label='Unexplained Soft Material')
    ax1.legend()

    # [證據 2] 深度 vs RL 的二維直方圖/散佈圖 (Topographic Pooling)
    # 使用 hexbin 來處理幾十萬個點的密集度
    hb = ax2.hexbin(rl_valid, bathy_valid, gridsize=50, cmap='viridis', mincnt=1)
    ax2.axvline(HAMILTON_LIMIT, color='red', linestyle='--', linewidth=2)

    ax2.set_title("Evidence 2: Topographic Pooling Effect", fontweight='bold')
    ax2.set_xlabel("Reflection Loss (dB)")
    ax2.set_ylabel("Bathymetry / Depth (m)")
    ax2.invert_yaxis()  # 深度越深往下

    cb = fig.colorbar(hb, ax=ax2)
    cb.set_label('Point Density')

    plt.tight_layout()
    # plt.savefig(OUT_PLOT, dpi=300)
    plt.show()
    # print(f"圖表已儲存至: {OUT_PLOT}")


if __name__ == "__main__":
    main()