# src/backscatter/correction.py
"""
SSS backscatter Angular Response Correction (ARC).

Workflow:
  1. Build global median ARC from all collected data
  2. Apply global correction to reference angle (45°)
  3. K-means clustering on ping-level features
  4. Per-cluster residual ARC correction via median-bin interpolation (AVG)
  5. Normalize all samples to reference angle

Note: Correction uses median-bin interpolation, not polynomial fitting.
      _polyfit_with_rejection has been removed as it was unused.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt
from scipy.stats import mode
from sklearn.cluster import KMeans

from src.config import REFERENCE_ANGLE


def _to_db(bs_linear):
    return 10 * np.log10(np.maximum(bs_linear, 1e-12).astype(np.float64))


def build_global_arc(bs_db, inc_angle, bins=np.arange(15, 86, 1)):
    """Build a single global median ARC from all collected data."""
    bin_centers = (bins[:-1] + bins[1:]) / 2
    curve = np.full(len(bins) - 1, np.nan)
    for j in range(len(bins) - 1):
        mask = (inc_angle >= bins[j]) & (inc_angle < bins[j+1]) & np.isfinite(bs_db)
        if mask.sum() >= 10:
            curve[j] = np.median(bs_db[mask])
    valid = np.isfinite(curve)
    if valid.sum() > 3:
        curve[valid] = uniform_filter1d(curve[valid], size=5)
    return bin_centers, curve


def apply_global_correction(bs_db, inc_angle, bin_centers, curve, ref_angle=REFERENCE_ANGLE):
    """Normalize BS to reference angle using global ARC."""
    valid = np.isfinite(curve)
    bs0 = np.interp(ref_angle, bin_centers[valid], curve[valid])
    correction = np.interp(inc_angle, bin_centers[valid], curve[valid])
    return bs_db - correction + bs0


def _smooth_labels(ping_labels, window=7):
    """Majority vote over a sliding window along track."""
    smoothed = np.array(ping_labels, dtype=np.int8)
    half = window // 2
    for i in range(len(ping_labels)):
        lo = max(0, i - half)
        hi = min(len(ping_labels), i + half + 1)
        smoothed[i] = mode(ping_labels[lo:hi], keepdims=False).mode
    return smoothed


def collect_features(bs_linear, inc_angle, ping_idx, bin_centers, arc_curve):
    """
    Build per-ping feature vectors from globally corrected BS.
    Each ping → one row: median BS in three angle bands.
    Falls back to slope extrapolation for narrow-swath pings missing outer band.
    Returns feature matrix and valid ping ids.
    """
    bs_db = _to_db(bs_linear)
    bs_corr = apply_global_correction(bs_db, inc_angle, bin_centers, arc_curve)

    angle_bands = [(15, 30), (30, 50), (50, 70)]

    df = pd.DataFrame({"ping": ping_idx, "bs": bs_corr, "inc": inc_angle})

    result_cols = []
    for lo, hi in angle_bands:
        band = df[(df["inc"] >= lo) & (df["inc"] < hi) & df["bs"].notna()]
        result_cols.append(band.groupby("ping")["bs"].median())

    feat_df = pd.concat(result_cols, axis=1)
    feat_df.columns = [0, 1, 2]

    # extrapolate missing outer band from slope of inner two bands
    nan2 = feat_df[2].isna()
    ok01 = feat_df[0].notna() & feat_df[1].notna()
    feat_df.loc[nan2 & ok01, 2] = feat_df.loc[nan2 & ok01, 1] * 2 - feat_df.loc[nan2 & ok01, 0]

    valid = feat_df.notna().sum(axis=1) >= 2
    feat_df = feat_df[valid]

    if len(feat_df) == 0:
        return None, None

    return feat_df.values.astype(np.float32), feat_df.index.values


def fit_kmeans(feature_list, n_clusters=7):
    """Fit K-means on collected ping-level features."""
    all_feat = np.concatenate(feature_list, axis=0)
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=0)
    km.fit(all_feat)
    return km


def predict_labels(km, feats, valid_pings, smooth_window=7):
    """Predict and smooth cluster labels for one line."""
    ping_labels = km.predict(feats).astype(np.int8)
    ping_labels = _smooth_labels(ping_labels, window=smooth_window)
    return valid_pings, ping_labels


def apply_arc_correction(bs_linear, inc_angle, ping_idx, valid_pings, ping_labels,
                         bin_centers, arc_curve, n_clusters=7, nadir_cutoff=10.0):
    """
    Apply per-cluster Angular Response Correction (ARC).

    Workflow:
      1. Global correction removes bulk angle effect
      2. Broadcast pre-computed smoothed ping labels to sample level
      3. Per-cluster median-bin ARC fitted and applied as residual correction
      4. All samples normalized to REFERENCE_ANGLE
    """
    bs_db = _to_db(bs_linear)
    bs_global = apply_global_correction(bs_db, inc_angle, bin_centers, arc_curve)

    # broadcast ping labels to sample level (vectorized)
    ping_to_label = np.full(int(ping_idx.max()) + 1, -1, dtype=np.int8)
    ping_to_label[valid_pings] = ping_labels
    labels = ping_to_label[ping_idx]

    bs_corr = bs_global.copy()
    bins = np.arange(nadir_cutoff, 66, 1)
    bin_centers_c = (bins[:-1] + bins[1:]) / 2

    # compute all cluster×bin medians in one groupby
    valid_mask = (inc_angle >= nadir_cutoff) & np.isfinite(bs_global) & (labels >= 0)

    if valid_mask.any():
        df = pd.DataFrame({
            "bs":    bs_global[valid_mask],
            "inc":   inc_angle[valid_mask],
            "label": labels[valid_mask],
        })
        df["bin_idx"] = pd.cut(df["inc"], bins=bins, labels=False, right=False)
        df = df.dropna(subset=["bin_idx"])
        df["bin_idx"] = df["bin_idx"].astype(np.int32)

        grouped  = df.groupby(["label", "bin_idx"])["bs"]
        medians  = grouped.median()
        counts   = grouped.count()

        for c in range(n_clusters):
            if c not in medians.index.get_level_values("label"):
                continue

            curve = np.full(len(bins) - 1, np.nan)
            c_med = medians[c]
            c_cnt = counts[c]
            valid_bins = c_cnt[c_cnt >= 5].index
            curve[valid_bins] = c_med[valid_bins].values

            valid = np.isfinite(curve)
            if valid.sum() == 0:
                continue
            if valid.sum() > 3:
                curve[valid] = uniform_filter1d(curve[valid], size=5)

            mask_c = (labels == c) & (inc_angle >= nadir_cutoff) & np.isfinite(bs_global)
            if not mask_c.any():
                continue

            bs0 = np.interp(REFERENCE_ANGLE, bin_centers_c[valid], curve[valid])
            correction = np.interp(inc_angle[mask_c], bin_centers_c[valid], curve[valid])
            bs_corr[mask_c] = bs_global[mask_c] - correction + bs0

    bs_corr[inc_angle < nadir_cutoff] = np.nan
    valid_corr = bs_corr[np.isfinite(bs_corr)]
    if len(valid_corr) > 0:
        q01, q99 = np.percentile(valid_corr, [1, 99])
        bs_corr = np.where(np.isfinite(bs_corr), np.clip(bs_corr, q01, q99), np.nan)

    return bs_corr, _to_db(bs_linear), labels


def detect_first_return(amps, pix_m, min_range_m=3.0, threshold_ratio=0.1):
    """Detect first bottom return using median-filtered envelope."""
    min_idx = int(min_range_m / pix_m)
    if min_idx >= len(amps):
        return None
    search = medfilt(amps[min_idx:].astype(np.float64), kernel_size=21)
    max_val = search.max()
    if max_val <= 0:
        return None
    threshold = max_val * threshold_ratio
    diff = np.diff(search)
    candidates = np.where(diff > threshold)[0]
    if len(candidates) == 0:
        return None
    return float((candidates[0] + min_idx) * pix_m)