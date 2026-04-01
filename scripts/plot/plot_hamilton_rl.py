import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import SOUND_SPEED

OUT_DIR = Path(__file__).parent.parent.parent / "outputs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEDIMENTS = {
    "Coarse sand":     (2034, 1836),
    "Fine sand":       (1962, 1759),
    "Very fine sand":  (1878, 1709),
    "Silty sand":      (1783, 1658),
    "Sandy silt":      (1769, 1644),
    "Silt":            (1740, 1615),
    "Sandy-silt-clay": (1575, 1582),
    "Clayey silt":     (1489, 1546),
    "Silty clay":      (1480, 1570),
}

RHO_W_MARINE = 1025.0
C_W_MARINE   = 1491.0

RHO_W_FRESH  = 1000.0
C_W_FRESH    = SOUND_SPEED


def calc_rl(rho_w, c_w, rho_s, c_s):
    z_w = rho_w * c_w
    z_s = rho_s * c_s
    R = (z_s - z_w) / (z_s + z_w)
    return -20 * np.log10(np.abs(R))


labels = list(SEDIMENTS.keys())
rl_marine = [calc_rl(RHO_W_MARINE, C_W_MARINE, s[0], s[1])
             for s in SEDIMENTS.values()]
rl_fresh  = [calc_rl(RHO_W_FRESH, C_W_FRESH, s[0], s[1])
             for s in SEDIMENTS.values()]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(labels, rl_marine, marker="o", linestyle="-", linewidth=2,
        markersize=8, label="Seawater (ρ=1025, c=1491)")
ax.plot(labels, rl_fresh, marker="s", linestyle="--", linewidth=2,
        markersize=8, label=f"Freshwater (ρ=1000, c={SOUND_SPEED})")

ax.set_ylabel("Reflection Loss (dB)", fontsize=12)
ax.set_title("Hamilton Table: Marine vs. Freshwater Adjusted RL", fontsize=14)
# ax.tick_params(axis="x", rotation=45)
ax.grid(True, linestyle=":", alpha=0.7)
ax.legend(fontsize=11)

for i in range(len(labels)):
    ax.annotate(f"{rl_marine[i]:.2f}", (i, rl_marine[i]),
                textcoords="offset points", xytext=(0, 10), fontsize=8)
    ax.annotate(f"{rl_fresh[i]:.2f}", (i, rl_fresh[i]),
                textcoords="offset points", xytext=(0, -17), fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "hamilton_rl_comparison.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'hamilton_rl_comparison.png'}")
plt.show()