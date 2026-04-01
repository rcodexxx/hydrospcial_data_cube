import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 若沒有 src.config，直接寫死你實測的淡水聲速
try:
    from src.config import SOUND_SPEED
except ImportError:
    SOUND_SPEED = 1489.11

OUT_DIR = Path(__file__).parent.parent.parent / "outputs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 注意：這裡已經把 Silty clay 和 Clayey silt 互換，確保 RL 遞增
SEDIMENTS = {
    "Coarse sand":     (2034, 1836),
    "Fine sand":       (1962, 1759),
    "Very fine sand":  (1878, 1709),
    "Silty sand":      (1783, 1658),
    "Sandy silt":      (1769, 1644),
    "Silt":            (1740, 1615),
    "Sandy-silt-clay": (1575, 1582),
    "Clayey silt":     (1489, 1546), # 13.37 dB
    "Silty clay":      (1480, 1570), # 13.20 dB
}

RHO_W_MARINE = 1025.0
C_W_MARINE   = 1491.0

RHO_W_FRESH  = 1000.0
C_W_FRESH    = SOUND_SPEED

# ---------------------------------------------------------
# 1. 物理公式定義區
# ---------------------------------------------------------
def calc_rl(rho_w, c_w, rho_s, c_s):
    """Hamilton 模型：壓實底質反射損失"""
    z_w = rho_w * c_w
    z_s = rho_s * c_s
    R = (z_s - z_w) / (z_s + z_w)
    return -20 * np.log10(np.abs(R))

def calc_wood_rl(rho_w, c_w, porosity=0.90):
    """Wood 方程式：懸浮流體泥反射損失"""
    rho_g = 2650.0
    c_g = 5000.0
    # 密度混合
    rho_mud = porosity * rho_w + (1 - porosity) * rho_g
    # 體積彈性模量 (K) 混合
    k_w = rho_w * (c_w**2)
    k_g = rho_g * (c_g**2)
    inv_k_mud = (porosity / k_w) + ((1 - porosity) / k_g)
    # 聲速反轉
    c_mud = np.sqrt(1 / (rho_mud * inv_k_mud))
    # 阻抗與 RL
    z_w = rho_w * c_w
    z_mud = rho_mud * c_mud
    R = (z_mud - z_w) / (z_mud + z_w)
    return -20 * np.log10(np.abs(R))

# ---------------------------------------------------------
# 2. 數據計算區
# ---------------------------------------------------------
labels = list(SEDIMENTS.keys())
rl_marine = [calc_rl(RHO_W_MARINE, C_W_MARINE, s[0], s[1]) for s in SEDIMENTS.values()]
rl_fresh  = [calc_rl(RHO_W_FRESH, C_W_FRESH, s[0], s[1]) for s in SEDIMENTS.values()]

# 加入 Fluid Mud (n=90%) 的計算點
labels.append("Fluid mud")
rl_marine.append(calc_wood_rl(RHO_W_MARINE, C_W_MARINE, 0.90))
rl_fresh.append(calc_wood_rl(RHO_W_FRESH, C_W_FRESH, 0.90))

# 取得淡水的絕對閾值
hamilton_limit = max(rl_fresh[:-1])  # Clayey silt 的 13.37 dB
fluid_mud_limit = rl_fresh[-1] # Fluid mud 的 23.95 dB

# ---------------------------------------------------------
# 3. 繪圖區
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.5))

# 使用水平色塊 (axhspan) 來呈現由 RL (Y軸) 決定的物理狀態區間

# 1. 壓實底質區 (Consolidated Sediments)
# 範圍：從 0 dB 到 Hamilton 的淡水極限 (13.37 dB)
ax.axhspan(0, hamilton_limit, color='lightgray', alpha=0.3,
           label='Consolidated Sediments')

# 2. 過渡軟泥區 (Framework-supported Mud)
# 範圍：從 Hamilton 極限 (13.37 dB) 到流體泥分界線 (~23.95 dB)
ax.axhspan(hamilton_limit, fluid_mud_limit, color='moccasin', alpha=0.4,
           label='Framework-supported Mud')

# 3. 流體泥區 (Fluid Mud)
# 範圍：大於流體泥分界線 (RL > ~23.95 dB)
ax.axhspan(fluid_mud_limit, 30, color='lightblue', alpha=0.4,
           label='Fluid Mud')

# ---------------------------------------

# 繪製主折線圖
# ax.plot(labels, rl_marine, marker="o", linestyle="-", linewidth=2,
#         markersize=8, label="Seawater (rho=1025, c=1491)", color='#d73027')
ax.plot(labels, rl_fresh, marker="s", linestyle="--", linewidth=2,
        markersize=8, label=f"Freshwater", color='#4575b4')

# 設定圖表標籤
ax.set_ylabel("Reflection Loss (dB)", fontsize=12, fontweight='bold')
ax.set_title("Extended Acoustic Reflection Loss Model", fontsize=14)
ax.grid(True, linestyle=":", alpha=0.6)

# 調整 X 軸文字
# ax.tick_params(axis="x", rotation=30, labelsize=10)
ax.set_ylim(5, 27)

# 整理圖例
handles, legends = ax.get_legend_handles_labels()
ax.legend(handles, legends, fontsize=10, loc='lower right', framealpha=0.95)

# 標註淡水數值
# for i in range(len(labels)):
#     offset = (0, -15) if i == len(labels)-1 else (0, 10)
#     ax.annotate(f"{rl_fresh[i]:.2f}", (i, rl_fresh[i]),
#                 textcoords="offset points", xytext=offset, fontsize=9,
#                 ha='center', fontweight='bold', color='#4575b4')

plt.tight_layout()
plt.savefig(OUT_DIR / "hamilton_wood_rl_comparison.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'hamilton_wood_rl_comparison.png'}")
plt.show()