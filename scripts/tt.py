import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 引入你原本的讀取模組
from src.backscatter.georef import georef_line
from src.config import ROOT

# ─── 請這裡替換成你水庫中「最具代表性（有平坦也有邊坡）」的一條 JSF 測線 ───
TEST_JSF = ROOT / "data/sss/20251223/20251223054335.jsf"
MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
CHANNEL = "HF_port"  # 先單獨看左舷


def main():
    if not TEST_JSF.exists():
        print(f"找不到檔案: {TEST_JSF}")
        return

    print(f"正在讀取最原始的 JSF 數據: {TEST_JSF.name}")
    # 完全跳過 Pass 2 和 Pass 3，只做最基礎的讀取與幾何定位
    r = georef_line(TEST_JSF, MBES_TIF, CHANNEL, cable_length=15.0, turn_threshold=5)

    if r is None:
        print("讀取失敗或測線無效。")
        return

    # 提取原始的線性 BS 數值與入射角
    raw_bs_linear = r["bs"]
    inc_angle = r["inc_angle"]

    # 過濾無效值並轉換為 dB (參考你 build_backscatter.py 裡的轉換邏輯)
    valid_mask = (raw_bs_linear > 0) & np.isfinite(inc_angle)
    raw_bs_db = 10 * np.log10(raw_bs_linear[valid_mask].astype(np.float64))
    inc_angle = inc_angle[valid_mask]

    print(f"取得 {len(raw_bs_db)} 筆有效聲學樣本。開始計算 Raw ARC...")

    # 計算每一個角度 (1度為一個 Bin) 的中位數 dB
    bins = np.arange(15, 86, 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    arc_curve = np.full(len(bins) - 1, np.nan)

    for i in range(len(bins) - 1):
        in_bin = (inc_angle >= bins[i]) & (inc_angle < bins[i + 1])
        if in_bin.sum() > 10:  # 確保該角度有足夠的樣本數
            arc_curve[i] = np.median(raw_bs_db[in_bin])

    # 繪製最原始的 ARC 曲線
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, arc_curve, marker='o', markersize=4, linestyle='-', color='dodgerblue', linewidth=2)

    plt.title(f"Raw Angular Response Curve\n{TEST_JSF.name} ({CHANNEL})", fontsize=14, fontweight='bold')
    plt.xlabel("Incidence Angle (degrees)", fontsize=12)
    plt.ylabel("Raw Backscatter Strength (dB)", fontsize=12)
    plt.xlim(10, 90)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 標示出 45 度參考線
    plt.axvline(x=45, color='red', linestyle=':', label='45° Sweet Spot')
    plt.legend()

    out_fig = ROOT / f"outputs/figures/raw_arc_check.png"
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    print(f"\n圖表已儲存至: {out_fig}")
    plt.show()


if __name__ == "__main__":
    main()