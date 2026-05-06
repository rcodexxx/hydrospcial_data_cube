"""
Zhao (2017) radiometric correction for SSS backscatter.

Pipeline:
  0. Gain normalization per (line, channel)        — Stage 2b
  1. Along-track TVG residual: bs *= h_0 / altitude
  2. Convert to dB
  3. Build angular waterfall (ping x angle_bin)
  4. Robust z-score per angle column (median + MAD)
  5. K-means clustering with iterative merge (Zhao §3.2.2)
  6. Mode filter on label image
  7. Per-cluster ABC: ABC(c, phi) = mean(bs_db | cluster=c, angle=phi)
  8. Apply Zhao Eq.8: bs_corrected = bs_raw - ABC(c, phi) + BS_0(c)
"""
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt2d
from sklearn.cluster import KMeans

from src.sss.config import (
    REFERENCE_ANGLE_DEG,
    NADIR_CUTOFF_DEG,
    get_far_cutoff,
    ANGLE_MIN_DEG, ANGLE_MAX_DEG, ANGLE_BIN_WIDTH_DEG,
    MIN_SAMPLES_PER_ANGLE_BIN, MIN_SAMPLES_PER_CLUSTER_BIN,
    MODE_FILTER_WINDOW,
    GAIN_NORM_PERCENTILE, GAIN_NORM_MIN_SAMPLES,
)


# ──────────────────────────────────────────────────────────────
# Gain normalization (Stage 2b)
# ──────────────────────────────────────────────────────────────
def normalize_gain(pooled, ref_percentile=GAIN_NORM_PERCENTILE):
    """
    AGC compensation: normalize bs_linear per (line, channel) by
    `ref_percentile`-th percentile.

    EdgeTech's per-ping dynamic AGC produces up to 50x amplitude
    differences across survey lines. This places all lines on a
    comparable linear scale before Zhao's radiometric correction.

    Returns a new pooled dict with normalized bs_linear.
    """
    pooled = {k: v.copy() if hasattr(v, "copy") else v
              for k, v in pooled.items()}
    bs_lin = pooled["bs_linear"]

    refs = []
    n_groups = 0
    for line in np.unique(pooled["line_id"]):
        for ch in (0, 1):
            mask = (pooled["line_id"] == line) & (pooled["channel_id"] == ch)
            if mask.sum() < GAIN_NORM_MIN_SAMPLES:
                continue
            ref = np.percentile(bs_lin[mask], ref_percentile)
            if ref > 0:
                bs_lin[mask] = bs_lin[mask] / ref
                refs.append(float(ref))
                n_groups += 1

    print(f"  Normalized {n_groups} (line, channel) groups")
    if refs:
        refs = np.array(refs)
        print(f"  Ref range: p5={np.percentile(refs, 5):.3f}, "
              f"median={np.median(refs):.3f}, "
              f"p95={np.percentile(refs, 95):.3f}")
        print(f"  Max/min ratio: {refs.max() / refs.min():.1f}x")

    return pooled


# ──────────────────────────────────────────────────────────────
# Along-track TVG residual (Zhao Eq.2)
# ──────────────────────────────────────────────────────────────
def altitude_correct(bs_linear, altitude, h0=None, clip_percentile=99.5,
                     ratio_min=0.5, ratio_max=2.0):
    """Zhao Eq.(2): bs *= h_0 / altitude. Clips extreme amplitudes."""
    if h0 is None:
        h0 = float(np.nanmedian(altitude))

    threshold = np.percentile(bs_linear[bs_linear > 0], clip_percentile)
    bs_clipped = np.minimum(bs_linear, threshold)

    ratio = np.clip(h0 / altitude, ratio_min, ratio_max)
    corrected = bs_clipped * ratio.astype(np.float32)
    return corrected.astype(np.float32), h0


# ──────────────────────────────────────────────────────────────
# Angular waterfall
# ──────────────────────────────────────────────────────────────
def build_angular_waterfall(bs_db, inc_angle, ping_idx):
    """Per-sample arrays -> 2D (n_pings x n_angle_bins) median grid."""
    bins = np.arange(ANGLE_MIN_DEG, ANGLE_MAX_DEG + ANGLE_BIN_WIDTH_DEG,
                     ANGLE_BIN_WIDTH_DEG)
    n_bins = len(bins) - 1
    angle_centers = ((bins[:-1] + bins[1:]) / 2).astype(np.float32)

    valid = (np.isfinite(bs_db) &
             (inc_angle >= ANGLE_MIN_DEG) &
             (inc_angle <  ANGLE_MAX_DEG))

    df = pd.DataFrame({
        "ping": ping_idx[valid],
        "bin":  np.digitize(inc_angle[valid], bins) - 1,
        "bs":   bs_db[valid],
    })
    df = df[(df["bin"] >= 0) & (df["bin"] < n_bins)]

    grouped = df.groupby(["ping", "bin"])["bs"].median().unstack("bin")
    grouped = grouped.reindex(columns=range(n_bins))

    ping_ids = grouped.index.values
    waterfall = grouped.values.astype(np.float32)
    return waterfall, angle_centers, ping_ids


# ──────────────────────────────────────────────────────────────
# Robust z-score
# ──────────────────────────────────────────────────────────────
def robust_zscore_per_angle(waterfall):
    """z = (x - median) / (1.4826 * MAD), per angle column."""
    z = np.full_like(waterfall, np.nan, dtype=np.float32)
    for j in range(waterfall.shape[1]):
        col = waterfall[:, j]
        valid = np.isfinite(col)
        if valid.sum() < 10:
            continue
        med = np.median(col[valid])
        mad = np.median(np.abs(col[valid] - med))
        if mad < 1e-6:
            continue
        z[valid, j] = (col[valid] - med) / (1.4826 * mad)
    return z


# ──────────────────────────────────────────────────────────────
# K-means with iterative merge (Zhao §3.2.2)
# ──────────────────────────────────────────────────────────────
def _kmeans_with_merge(features, k_init, bs_range,
                       min_proportion=0.01, merge_distance_ratio=0.02):
    """K-means++ with iterative merge for tight clusters."""
    k = k_init
    labels = None
    merge_threshold = merge_distance_ratio * bs_range

    for _ in range(10):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0)
        labels = km.fit_predict(features).astype(np.int16)
        centers = km.cluster_centers_.ravel()

        unique, counts = np.unique(labels, return_counts=True)
        total = counts.sum()
        keep_mask = (counts / total) >= min_proportion
        keep_ids = unique[keep_mask]
        if len(keep_ids) == 0:
            break

        keep_centers = centers[keep_ids]
        order = np.argsort(keep_centers)
        sorted_ids = keep_ids[order]
        sorted_centers = keep_centers[order]

        merged_ids = [sorted_ids[0]]
        for m in range(1, len(sorted_ids)):
            if sorted_centers[m] - sorted_centers[m - 1] < merge_threshold:
                continue
            merged_ids.append(sorted_ids[m])

        new_k = len(merged_ids)
        if new_k == k:
            break
        k = new_k

    return labels


def run_clustering(zscore_waterfall, cluster_cfg):
    """Dispatch clustering based on yaml config."""
    valid = np.isfinite(zscore_waterfall)
    features = zscore_waterfall[valid].reshape(-1, 1)

    print(f"  z-score features: n={len(features)}, "
          f"p5={np.percentile(features, 5):.2f}, "
          f"p50={np.median(features):.2f}, "
          f"p95={np.percentile(features, 95):.2f}, "
          f"std={features.std():.2f}")

    method = cluster_cfg["method"]
    if method == "kmeans":
        bs_range = float(features.max() - features.min())
        labels_flat = _kmeans_with_merge(
            features,
            k_init=cluster_cfg.get("k_init", 7),
            bs_range=bs_range,
        )
    elif method == "hdbscan":
        import hdbscan
        n_valid = features.shape[0]
        min_frac = cluster_cfg["min_cluster_fraction"]
        min_cluster_size = max(int(n_valid * min_frac), 10)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=cluster_cfg["min_samples"],
            core_dist_n_jobs=-1,
        )
        labels_flat = clusterer.fit_predict(features).astype(np.int16)
    else:
        raise ValueError(f"Unknown cluster method: {method}")

    labels = np.full(zscore_waterfall.shape, -1, dtype=np.int16)
    labels[valid] = labels_flat
    return labels


def smooth_labels(labels):
    """2D mode filter on label image to suppress single-pixel speckle."""
    if MODE_FILTER_WINDOW <= 1:
        return labels
    smoothed = medfilt2d(labels.astype(np.int16), kernel_size=MODE_FILTER_WINDOW)
    smoothed[labels == -1] = -1
    return smoothed


# ──────────────────────────────────────────────────────────────
# ABC curves
# ──────────────────────────────────────────────────────────────
def compute_per_cluster_abc(waterfall, labels):
    """Per-cluster ABC: ABC(c, phi) = mean(bs_db | cluster=c, angle=phi)."""
    cluster_ids = np.unique(labels)
    cluster_ids = cluster_ids[cluster_ids >= 0]

    abc_by_cluster = {}
    n_bins = waterfall.shape[1]

    for c in cluster_ids:
        curve = np.full(n_bins, np.nan, dtype=np.float32)
        for j in range(n_bins):
            mask = (labels[:, j] == c) & np.isfinite(waterfall[:, j])
            if mask.sum() >= MIN_SAMPLES_PER_CLUSTER_BIN:
                curve[j] = np.mean(waterfall[mask, j])

        valid = np.isfinite(curve)
        if valid.sum() < 3:
            continue
        curve[valid] = uniform_filter1d(curve[valid], size=5)
        bs_0 = float(np.nanmean(curve))
        abc_by_cluster[int(c)] = (curve, bs_0)

    return abc_by_cluster


def apply_per_cluster_abc(bs_db, inc_angle, ping_idx, sample_labels,
                          abc_by_cluster, angle_centers):
    """Apply Zhao Eq.8 per sample using its cluster's ABC."""
    bs_corr = bs_db.copy()
    for c, (abc, bs_0) in abc_by_cluster.items():
        mask = sample_labels == c
        if not mask.any():
            continue
        valid = np.isfinite(abc)
        if valid.sum() < 2:
            continue
        correction = np.interp(inc_angle[mask], angle_centers[valid], abc[valid])
        bs_corr[mask] = bs_db[mask] - correction + bs_0
    return bs_corr.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────
def to_db(bs_linear):
    """
    Convert linear backscatter to dB.

    Samples with bs_linear ≤ 1e-12 (typically dropped/saturated raw
    samples that EdgeTech firmware writes as 0) become NaN so they
    skip mosaic accumulation entirely. Without this, np.maximum
    floors them to -120 dB and pollutes cell averages.
    """
    result = np.full_like(bs_linear, np.nan, dtype=np.float32)
    valid = bs_linear > 1e-12
    result[valid] = 10.0 * np.log10(bs_linear[valid])
    return result


def mask_out_of_range_angles(bs_db, inc_angle, freq):
    """Set samples outside [NADIR_CUTOFF, FAR_CUTOFF(freq)] to NaN."""
    far_cutoff = get_far_cutoff(freq)
    bad = (inc_angle < NADIR_CUTOFF_DEG) | (inc_angle > far_cutoff)
    bs_db = bs_db.copy()
    bs_db[bad] = np.nan
    return bs_db


def run_correction(pooled, cluster_cfg, freq):
    """
    Full Zhao correction on pooled samples.

    pooled: dict with bs_linear, altitude, inc_angle, ping_idx,
            line_id, channel_id (all 1D, equal length).
    freq:   'HF' or 'LF' — selects channel-aware angle cutoff.

    Returns: dict with bs_db, sample_labels, diagnostics.
    """
    bs_lin = pooled["bs_linear"].copy()
    alt = pooled["altitude"]
    inc = pooled["inc_angle"]
    pid = pooled["ping_idx"]
    diag = {}

    # Step 1: along-track TVG residual
    bs_lin, h0 = altitude_correct(bs_lin, alt)
    diag["h0"] = h0
    ratio = h0 / alt
    print(f"  correction ratio: min={ratio.min():.2f}, "
          f"median={np.median(ratio):.2f}, max={ratio.max():.2f}")

    # Step 2: dB
    bs_db = to_db(bs_lin)

    # Step 3: angular waterfall
    waterfall, angle_centers, ping_ids = build_angular_waterfall(bs_db, inc, pid)
    diag["waterfall_shape"] = waterfall.shape
    diag["angle_centers"] = angle_centers
    diag["waterfall"] = waterfall

    # Step 4-5: z-score + cluster
    zscore = robust_zscore_per_angle(waterfall)
    labels_2d = run_clustering(zscore, cluster_cfg)
    labels_2d = smooth_labels(labels_2d)
    diag["labels_2d"] = labels_2d
    diag["ping_ids"] = ping_ids
    diag["n_clusters"] = (int(labels_2d[labels_2d >= 0].max() + 1)
                          if (labels_2d >= 0).any() else 0)
    diag["noise_ratio"] = float((labels_2d == -1).sum() / labels_2d.size)

    # Broadcast ping-bin labels to sample level
    bin_edges = np.arange(ANGLE_MIN_DEG,
                          ANGLE_MAX_DEG + ANGLE_BIN_WIDTH_DEG,
                          ANGLE_BIN_WIDTH_DEG)
    sample_bin = np.digitize(inc, bin_edges) - 1
    in_range = (sample_bin >= 0) & (sample_bin < labels_2d.shape[1])
    row_idx = np.searchsorted(ping_ids, pid)
    row_idx = np.clip(row_idx, 0, len(ping_ids) - 1)
    row_valid = ping_ids[row_idx] == pid

    sample_labels = np.full(len(bs_db), -1, dtype=np.int16)
    final_valid = in_range & row_valid
    sample_labels[final_valid] = labels_2d[
        row_idx[final_valid], sample_bin[final_valid]
    ]

    # Step 7-8: per-cluster ABC + apply
    abc_by_cluster = compute_per_cluster_abc(waterfall, labels_2d)
    diag["abc_by_cluster"] = abc_by_cluster

    bs_corrected = apply_per_cluster_abc(
        bs_db, inc, pid, sample_labels, abc_by_cluster, angle_centers)

    bs_corrected = mask_out_of_range_angles(bs_corrected, inc, freq)

    return {"bs_db": bs_corrected, "sample_labels": sample_labels,
            "diagnostics": diag}