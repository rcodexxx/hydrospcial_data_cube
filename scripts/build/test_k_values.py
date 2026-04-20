# scripts/build/test_k_values.py
"""
Standalone script to test different K values (4~12) for SSS Angular Response Clustering.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
from src.backscatter.georef import georef_line
from src.backscatter.correction import collect_features, build_global_arc
from src.config import ROOT

CHANNELS = ["HF_port", "HF_stbd"]
MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
SURVEY_DIRS = {
    ROOT / "data/sss/20251223": 15.0,
    ROOT / "data/sss/20251224": None,
}
K_MIN = 4
K_MAX = 12
OUT_FIG = ROOT / "outputs/figures/k_value_comprehensive_test.png"


def main():
    jsf_files = [(f, cable)
                 for d, cable in SURVEY_DIRS.items()
                 for f in sorted(d.glob("*.jsf"))]
    print(f"Found {len(jsf_files)} JSF files for testing.")

    # ── Pass 1: 收集raw BS建global ARC ──────────────────────
    print("\nPass 1: Building global ARC...")
    raw_bs_all, raw_inc_all, all_results = [], [], []

    for jsf_path, cable in tqdm(jsf_files, desc="Reading raw BS"):
        for ch in CHANNELS:
            r = georef_line(jsf_path, MBES_TIF, ch, cable_length=cable,
                            turn_threshold=3, turn_cooldown=20.0)
            if r is None:
                continue
            bs_db = 10 * np.log10(np.maximum(r["bs"], 1e-12))
            raw_bs_all.append(bs_db)
            raw_inc_all.append(r["inc_angle"])
            all_results.append(r)

    if not all_results:
        print("Error: No valid data found.")
        return

    bin_centers_arc, arc_curve = build_global_arc(
        np.concatenate(raw_bs_all),
        np.concatenate(raw_inc_all)
    )
    print("Global ARC built.")

    # ── Pass 2: 建ping-level features + 收集sample-level繪圖資料 ──
    print("\nPass 2: Building ping-level features...")
    sample_feats = []
    all_bs_list, all_inc_list, all_ping_id_list = [], [], []
    feats_global_ping_ids = []
    global_ping_offset = 0

    for r in tqdm(all_results, desc="Building features"):
        feat, valid_pings = collect_features(
            r["bs"], r["inc_angle"], r["ping_idx"], bin_centers_arc, arc_curve
        )
        if feat is not None:
            sample_feats.append(feat)
            feats_global_ping_ids.extend(valid_pings + global_ping_offset)

        # 抽樣控制記憶體，保留ping_id對應關係
        n = len(r["bs"])
        idx = np.random.choice(n, min(n, 5000), replace=False)
        valid = r["bs"][idx] > 0
        all_bs_list.append(r["bs"][idx][valid])
        all_inc_list.append(r["inc_angle"][idx][valid])
        all_ping_id_list.append(r["ping_idx"][idx][valid] + global_ping_offset)
        global_ping_offset += int(r["ping_idx"].max()) + 1

    if not sample_feats:
        print("Error: No valid features found.")
        return

    all_feats = np.concatenate(sample_feats)
    all_bs = 10 * np.log10(np.maximum(np.concatenate(all_bs_list), 1e-12))
    all_inc = np.concatenate(all_inc_list)
    all_ping_ids = np.concatenate(all_ping_id_list)
    feats_global_ping_ids = np.array(feats_global_ping_ids, dtype=np.int64)

    print(f"Total pings with valid features: {len(all_feats):,}")
    print(f"Total sample points for ARC:     {len(all_bs):,}")

    # ── K-means測試 + 繪圖 ───────────────────────────────────
    print(f"\nRunning MiniBatchKMeans for K = {K_MIN} to {K_MAX}...")
    k_values = list(range(K_MIN, K_MAX + 1))
    inertias = []

    fig = plt.figure(figsize=(24, 16))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    bins = np.arange(15, 76, 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    cmap = plt.get_cmap('tab20')

    for i, k in enumerate(k_values):
        print(f"  Testing K = {k:2d} ...")
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
        ping_labels = km.fit_predict(all_feats)
        inertias.append(km.inertia_)

        # ping_id → cluster label 查找表
        label_lookup = dict(zip(feats_global_ping_ids.tolist(), ping_labels.tolist()))

        # 廣播到sample level
        sample_labels = np.array(
            [label_lookup.get(int(pid), -1) for pid in all_ping_ids],
            dtype=np.int32
        )

        ax = fig.add_subplot(3, 4, i + 2)

        for c in range(k):
            mask_c = (sample_labels == c)
            curve = np.full(len(bins) - 1, np.nan)

            for j in range(len(bins) - 1):
                in_bin = mask_c & (all_inc >= bins[j]) & (all_inc < bins[j + 1])
                if in_bin.sum() >= 5:
                    curve[j] = np.median(all_bs[in_bin])

            valid = np.isfinite(curve)
            if valid.sum() > 3:
                curve[valid] = uniform_filter1d(curve[valid], size=3)

            ax.plot(bin_centers, curve, color=cmap(c % 20), linewidth=2.5, alpha=0.9)

        ax.set_title(f"ARC for K = {k}", fontsize=14, fontweight='bold')
        ax.set_xlim(15, 85)
        ax.set_xlabel("Incidence Angle (deg)")
        ax.set_ylabel("BS (dB)")
        ax.grid(True, linestyle='--', alpha=0.5)

    # ── Elbow ────────────────────────────────────────────────
    ax_elbow = fig.add_subplot(3, 4, 1)
    ax_elbow.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax_elbow.set_title("Elbow Method (Inertia)", fontsize=14, fontweight='bold')
    ax_elbow.set_xlabel("Number of Clusters (K)")
    ax_elbow.set_ylabel("Sum of Squared Distances")
    ax_elbow.set_xticks(k_values)
    ax_elbow.grid(True, linestyle='--', alpha=0.5)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {OUT_FIG}")


if __name__ == "__main__":
    main()