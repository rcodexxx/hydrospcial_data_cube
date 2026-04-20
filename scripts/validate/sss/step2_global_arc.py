# scripts/validate/step2_global_arc.py
"""Check global ARC shape from all test JSF files."""
import numpy as np
import matplotlib.pyplot as plt
from src.backscatter.georef import georef_line
from src.backscatter.correction import build_global_arc
from src.config import ROOT

MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
OUT = ROOT / "outputs/figures/validate_step2_global_arc.png"

raw_bs_all, raw_inc_all = [], []
for jsf in sorted((ROOT / r"data\sss\20251223").glob("*.jsf")):
    for ch in ["HF_port", "HF_stbd"]:
        r = georef_line(jsf, MBES_TIF, ch, cable_length=None,
                        turn_threshold=3, turn_cooldown=20.0)
        if r is None:
            continue
        bs_db = 10 * np.log10(np.maximum(r["bs"], 1e-12))
        raw_bs_all.append(bs_db)
        raw_inc_all.append(r["inc_angle"])

bin_centers, arc_curve = build_global_arc(
    np.concatenate(raw_bs_all),
    np.concatenate(raw_inc_all)
)

plt.figure(figsize=(8, 5))
plt.plot(bin_centers, arc_curve, linewidth=2)
plt.axvline(45, color="red", linestyle="--", label="Reference angle (45°)")
plt.xlabel("Incidence Angle (deg)")
plt.ylabel("BS (dB)")
plt.title("Global ARC")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")