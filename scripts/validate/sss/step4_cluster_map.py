# scripts/validate/step4_cluster_map.py
"""Check spatial distribution of K-means cluster labels."""
import numpy as np
import matplotlib.pyplot as plt
from src.backscatter.georef import georef_line
from src.backscatter.correction import (
    build_global_arc, collect_features, fit_kmeans, predict_labels
)
from src.config import ROOT

MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
OUT = ROOT / "outputs/figures/validate_step4_cluster_map.png"
N_CLUSTERS = 5

raw_bs_all, raw_inc_all, all_results = [], [], []
jsf_list = sorted((ROOT / "data/sss/20251223").glob("*.jsf"))[:5]
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

sample_feats = []
for r in all_results:
    feat, _ = collect_features(r["bs"], r["inc_angle"], r["ping_idx"],
                                bin_centers, arc_curve)
    if feat is not None:
        sample_feats.append(feat)

km = fit_kmeans(sample_feats, N_CLUSTERS)

plt.figure(figsize=(10, 8))
cmap = plt.get_cmap("tab10")

for r in all_results:
    feat, valid_pings = collect_features(r["bs"], r["inc_angle"], r["ping_idx"],
                                          bin_centers, arc_curve)
    if feat is None:
        continue
    valid_pings, ping_labels = predict_labels(km, feat, valid_pings)
    for pid, lbl in zip(valid_pings, ping_labels):
        mask = r["ping_idx"] == pid
        if mask.any():
            plt.scatter(r["lon"][mask][0], r["lat"][mask][0],
                       color=cmap(lbl % 10), s=3, alpha=0.7)

# 畫cluster的色票
for c in range(N_CLUSTERS):
    plt.scatter([], [], color=cmap(c % 10), label=f"Cluster {c}", s=20)
plt.legend(markerscale=3, loc="upper right")
plt.xlabel("Lon")
plt.ylabel("Lat")
plt.title("Cluster Spatial Distribution (one point per ping)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")