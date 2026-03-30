import numpy as np
import matplotlib.pyplot as plt

# 沉積物物理性質 (Hamilton 1970 測定之固體基準): (密度 kg/m3, 聲速 m/s)
SEDIMENTS = {
    "Coarse sand":    (2034, 1836),
    "Fine sand":      (1962, 1759),
    "Very fine sand": (1878, 1709),
    "Silty sand":     (1783, 1658),
    "Sandy silt":     (1769, 1644),
    "Silt": (1740, 1615),
    "Sandy-silt-clay": (1575, 1582),
    "Clayey silt":    (1489, 1546),
    "Silty clay":     (1480, 1570)
}

# 海水環境參數 (傳統 Hamilton 表格預設背景)
RHO_W_MARINE = 1025.0
C_W_MARINE = 1491.0

# 淡水水庫環境參數 (來自你的 SVP 底層數據)
RHO_W_FRESH = 1000.0
C_W_FRESH = 1489.11

def calc_rl(rho_w, c_w, rho_s, c_s):
    """計算反射損失 RL (dB)"""
    z_w = rho_w * c_w
    z_s = rho_s * c_s
    R = (z_s - z_w) / (z_s + z_w)
    return -20 * np.log10(np.abs(R))

labels = list(SEDIMENTS.keys())
rl_marine = [calc_rl(RHO_W_MARINE, C_W_MARINE, s[0], s[1]) for s in SEDIMENTS.values()]
rl_fresh = [calc_rl(RHO_W_FRESH, C_W_FRESH, s[0], s[1]) for s in SEDIMENTS.values()]

# 繪製折線圖
fig, ax = plt.subplots(figsize=(10, 6))

# 海水曲線
ax.plot(labels, rl_marine, marker='o', linestyle='-', linewidth=2, markersize=8,
        label='Seawater')

# 淡水曲線
ax.plot(labels, rl_fresh, marker='s', linestyle='--', linewidth=2, markersize=8,
        label='Freshwater')

ax.set_ylabel('Reflection Loss (dB)', fontsize=12)
ax.set_title('Hamilton Table: Marine vs. Freshwater Adjusted RL', fontsize=14)
# ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend(fontsize=11)

# 標註數值 (為了避免重疊，數值稍微錯開)
for i, txt in enumerate(rl_marine):
    ax.annotate(f"{txt:.2f}", (i, rl_marine[i]), textcoords="offset points", xytext=(0,10))

for i, txt in enumerate(rl_fresh):
    ax.annotate(f"{txt:.2f}", (i, rl_fresh[i]), textcoords="offset points", xytext=(0,-17))

plt.tight_layout()
# plt.savefig('hamilton_line_chart.png', dpi=300)
plt.show()