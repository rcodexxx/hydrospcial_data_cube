# scripts/build/test_k_values.py
"""
Test different K values for SSS ARC clustering.
Reads ping-level features from existing npz files.

Usage:
    python scripts/build/test_k_values.py --freq hf
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

from src.config import CFG, ROOT

K_MIN = 4
K_MAX = 8
SAMPLES_PER_FILE = 5000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", choices=["hf", "lf"], default="hf")
    args = parser.parse_args()

    sss_cfg       = CFG[f"sss_{args.freq}"]
    corrected_dir = ROOT / sss_cfg["outputs"]["corrected_dir"]
    nadir_cut     = sss_cfg["arc_correction"]["nadir_cutoff"]
    far_cut       = sss_cfg["arc_correction"]["far_cutoff"]
    npz_files     = sorted(corrected_dir.glob(f"*_{args.freq.upper()}*.npz"))

    if not npz_files:
        print(f"No npz files in {corrected_dir}")
        return

    # load ping-level features + sample-level data for ARC plotting
    print(f"Loading {len(npz_files)} npz files...")
    feat_all, bs_all, inc_all = [], [], []

    for npz_path in tqdm(npz_files, desc="Loading"):
        d = np.load(npz_path)
        for side in ("port", "stbd"):
            feat_key = f"{side}_ping_features"
            if feat_key in d and len(d[feat_key]) > 0:
                feat_all.append(d[feat_key])

            if f"{side}_bs_raw" not in d:
                continue
            inc = d[f"{side}_inc"]
            n   = len(inc)
            idx = np.random.choice(n, min(n, SAMPLES_PER_FILE), replace=False)
            mask = (inc[idx] >= nadir_cut) & (inc[idx] <= far_cut)
            bs_all.append(d[f"{side}_bs_raw"][idx][mask])
            inc_all.append(inc[idx][mask])

    if not feat_all:
        print("No ping features found in npz files.")
        print("Make sure correct.py saves ping_features.")
        return

    feat_matrix = np.concatenate(feat_all, axis=0)
    feat_matrix = feat_matrix[np.isfinite(feat_matrix).all(axis=1)]

    all_bs  = 10 * np.log10(np.maximum(np.concatenate(bs_all), 1e-12))
    all_inc = np.concatenate(inc_all)
    valid   = np.isfinite(all_bs) & np.isfinite(all_inc)
    bs_for_arc  = all_bs[valid]
    inc_for_arc = all_inc[valid]

    print(f"Ping features: {len(feat_matrix):,}")
    print(f"Sample points: {len(bs_for_arc):,}")

    bins        = np.arange(nadir_cut, far_cut + 1, 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    cmap        = plt.get_cmap("tab10")
    k_values    = list(range(K_MIN, K_MAX + 1))
    inertias, rms_list = [], []

    n_cols = 4
    n_rows = (len(k_values) + n_cols) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()

    print(f"\nTesting K = {K_MIN} to {K_MAX}...")
    for i, k in enumerate(k_values):
        km     = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(feat_matrix)
        inertias.append(km.inertia_)

        # broadcast ping labels to sample level via nearest centroid
        sample_labels = km.predict(feat_matrix[:len(inc_for_arc)] 
                                   if len(feat_matrix) >= len(inc_for_arc) 
                                   else feat_matrix)

        # use angle-band median per sample as proxy feature for prediction
        n_bands  = feat_matrix.shape[1]
        edges    = np.linspace(nadir_cut, far_cut, n_bands + 1)
        proxy    = np.full((len(inc_for_arc), n_bands), np.nan)
        for b in range(n_bands):
            m = (inc_for_arc >= edges[b]) & (inc_for_arc < edges[b + 1])
            proxy[m, b] = bs_for_arc[m]

        valid_rows = np.isfinite(proxy).any(axis=1)
        sample_labels = np.full(len(inc_for_arc), -1, dtype=np.int32)
        if valid_rows.any():
            sample_labels[valid_rows] = km.predict(
                np.nan_to_num(proxy[valid_rows], nan=0.0)
            )

        ax = axes[i + 1]
        cluster_rms = []

        for c in range(k):
            mask_c = (sample_labels == c)
            if mask_c.sum() < 10:
                continue
            curve = np.full(len(bins) - 1, np.nan)
            for j in range(len(bins) - 1):
                in_bin = mask_c & (inc_for_arc >= bins[j]) & (inc_for_arc < bins[j + 1])
                if in_bin.sum() >= 10:
                    curve[j] = np.median(bs_for_arc[in_bin])

            valid_c = np.isfinite(curve)
            if valid_c.sum() > 3:
                curve[valid_c] = uniform_filter1d(curve[valid_c], size=3)
                rms = np.sqrt(np.nanmean((curve[valid_c] - np.nanmean(curve[valid_c])) ** 2))
                cluster_rms.append(rms)
            ax.plot(bin_centers, curve, color=cmap(c % 10), linewidth=2, alpha=0.9)

        mean_rms = float(np.mean(cluster_rms)) if cluster_rms else np.nan
        rms_list.append(mean_rms)
        ax.set_title(f"K={k}  ARC RMS={mean_rms:.3f} dB", fontsize=11)
        ax.set_xlim(nadir_cut, far_cut)
        ax.set_xlabel("Incidence angle (deg)")
        ax.set_ylabel("BS (dB)")
        ax.grid(True, linestyle="--", alpha=0.4)
        print(f"  K={k:2d}  inertia={km.inertia_:.0f}  mean ARC RMS={mean_rms:.3f} dB")

    # elbow + RMS panel
    ax  = axes[0]
    ax2 = ax.twinx()
    ax.plot(k_values, inertias, "bo-", linewidth=2, markersize=7)
    ax2.plot(k_values, rms_list, "rs--", linewidth=2, markersize=7)
    ax.set_title("Elbow + ARC RMS", fontsize=11)
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia", color="blue")
    ax2.set_ylabel("Mean ARC RMS (dB)", color="red")
    ax.set_xticks(k_values)
    ax.grid(True, linestyle="--", alpha=0.4)

    for j in range(len(k_values) + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"K-value Test — SSS {args.freq.upper()}", fontsize=13)
    plt.tight_layout()

    out_fig = ROOT / f"outputs/figures/k_value_test_{args.freq}.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out_fig}")
    plt.show()


if __name__ == "__main__":
    main()