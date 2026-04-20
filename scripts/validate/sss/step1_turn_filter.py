# scripts/validate/step1_turn_filter.py
"""Check that turn filtering removes heading-change pings."""
import matplotlib.pyplot as plt
from pathlib import Path
from src.backscatter.georef import georef_line
from src.config import ROOT

JSF = next((ROOT / "data/test").glob("*.jsf"))
MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
OUT = ROOT / "outputs/figures/validate_step1_turn_filter.png"

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (thresh, cooldown) in zip(axes, [(5, 0.0), (3, 20.0)]):
    r = georef_line(JSF, MBES_TIF, "HF_port", cable_length=None,
                    turn_threshold=thresh, turn_cooldown=cooldown)
    if r is None:
        ax.set_title("No data")
        continue
    ax.scatter(r["lon"], r["lat"], c=r["heading"], cmap="hsv", s=0.5, alpha=0.5)
    ax.set_title(f"threshold={thresh}°  cooldown={cooldown}s\n{len(set(r['ping_idx']))} pings")
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")