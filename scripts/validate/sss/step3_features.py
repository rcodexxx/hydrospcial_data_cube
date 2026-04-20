# scripts/validate/step3_features.py
"""Check per-ping feature distributions after global correction."""
import numpy as np
import matplotlib.pyplot as plt
from src.backscatter.georef import georef_line
from src.backscatter.correction import build_global_arc, collect_features
from src.config import ROOT

MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
OUT = ROOT / "outputs/figures/validate_step3_features.png"

raw_bs_all, raw_inc_all, all_results = [], [], []
jsf_list = sorted((ROOT / "data/sss/20251223").glob("*.jsf"))[:3]
for jsf in jsf_list:
    for ch in ["HF_port", "HF_stbd"]:
        r = georef_line(jsf, MBES_TIF, ch, cable_length=None,
                        turn_threshold=3, turn_cooldown=20.0)
        if r is None:
            continue
        bs_db = 10 * np.log10(np.maximum(r["bs"], 1e-12))
        raw_bs_all.append(bs_db)
        raw_inc_all.append(r["inc_angle"])
        all_results.append(r)

bin_centers, arc_curve = build_global_arc(
    np.concatenate(raw_bs_all),
    np.concatenate(raw_inc_all)
)

all_feats = []
for r in all_results:
    feat, _ = collect_features(r["bs"], r["inc_angle"], r["ping_idx"],
                                bin_centers, arc_curve)
    if feat is not None:
        all_feats.append(feat)

all_feats = np.concatenate(all_feats)
print(f"Total pings with valid features: {len(all_feats)}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
band_labels = ["15-30°", "30-50°", "50-70°"]
for i, ax in enumerate(axes):
    ax.hist(all_feats[:, i], bins=40, edgecolor="black", linewidth=0.3)
    ax.set_title(f"Band {band_labels[i]}")
    ax.set_xlabel("BS (dB)")
    ax.set_ylabel("Ping count")
    ax.grid(True, linestyle="--", alpha=0.4)

plt.suptitle("Per-ping Feature Distribution (after global correction)")
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")