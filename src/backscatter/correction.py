# src/backscatter/correction.py
"""
SSS backscatter angular correction.

Angular correction workflow (Zhao 2017):
  1. Collect BS samples, build z-score features
  2. K-means clustering on angular response features
  3. Per-cluster polynomial fitting with outlier rejection
  4. Normalize all pixels to reference angle

dB convention: 10 * log10(amplitude)
  EdgeTech envelope is intensity-proportional (confirmed from proc_flags=1).
  No REFERENCE_DN normalization — angular correction uses difference
  operations, so absolute offset cancels out.
"""

import warnings

import numpy as np
from numpy.exceptions import RankWarning
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt
from sklearn.cluster import KMeans

from src.config import REFERENCE_ANGLE


def _to_db(bs_linear):
    """Convert linear amplitude to dB. Intensity domain: 10*log10."""
    return 10 * np.log10(np.maximum(bs_linear, 1e-12).astype(np.float64))


def _build_features(bs_db, inc_angle):
    """
    Build physics-based features for k-means.
    Instead of statistical z-score, we use 'Delta dB' (difference from the
    global median angular response). This removes the angle effect but preserves
    the absolute physical impedance differences!
    """
    bins = np.arange(0, 91, 1)
    delta_db = np.zeros_like(bs_db, dtype=np.float32)

    for i in range(len(bins) - 1):
        mask = (inc_angle >= bins[i]) & (inc_angle < bins[i + 1]) & np.isfinite(bs_db)
        if mask.sum() < 2:
            continue

        mu = np.median(bs_db[mask])
        delta_db[mask] = bs_db[mask] - mu

    return delta_db.reshape(-1, 1).astype(np.float32)


def collect_features(bs_linear, inc_angle, sample_ratio=0.01):
    """Sample and build features for k-means training with outlier rejection."""
    bs_db = _to_db(bs_linear)

    idx = np.random.choice(
        len(bs_db), max(1, int(len(bs_db) * sample_ratio)), replace=False
    )

    bs_sample = bs_db[idx]
    inc_sample = inc_angle[idx]

    p01, p99 = np.percentile(bs_sample, [1, 99])
    mask = (bs_sample >= p01) & (bs_sample <= p99)

    feat = _build_features(bs_sample[mask], inc_sample[mask])
    return feat, bs_sample[mask], inc_sample[mask]


def fit_kmeans(feature_list, n_clusters=7):
    """Fit k-means on collected angular response features."""
    all_feat = np.concatenate(feature_list, axis=0)
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=0)
    km.fit(all_feat)
    return km


def _polyfit_with_rejection(x, y, deg, n_iter=2, sigma_thresh=2.0):
    """Iterative polynomial fitting with outlier rejection."""
    mask = np.ones(len(x), dtype=bool)
    coeffs = None

    for iteration in range(n_iter + 1):
        if mask.sum() < deg + 2:
            break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RankWarning)
            coeffs = np.polyfit(x[mask], y[mask], deg)
        if iteration < n_iter:
            residual = y - np.polyval(coeffs, x)
            std = np.std(residual[mask])
            if std > 0:
                mask = np.abs(residual) < sigma_thresh * std

    return coeffs


def angle_correction(
    bs_linear, inc_angle, km, n_clusters=7, nadir_cutoff=10.0, poly_deg=3
):
    """
    Apply angular correction using k-means clusters and polynomial fitting.

    Returns:
      bs_corr: corrected BS in dB (normalized to REFERENCE_ANGLE)
      bs_raw:  uncorrected BS in dB
      labels:  k-means cluster labels
    """
    # convert to dB (same convention as collect_features)
    valid_amp = bs_linear > 0
    bs_raw = np.full(len(bs_linear), np.nan, dtype=np.float64)
    bs_raw[valid_amp] = _to_db(bs_linear[valid_amp])

    # build features and predict clusters
    feat = _build_features(bs_raw, inc_angle)
    labels = km.predict(feat).astype(np.int8)
    bs_corr = bs_raw.copy()

    # per-cluster angular correction
    for c in range(n_clusters):
        mask = (labels == c) & (inc_angle >= nadir_cutoff) & np.isfinite(bs_raw)
        if mask.sum() < poly_deg + 2:
            continue

        bins = np.arange(nadir_cutoff, 86, 1)  # 10° to 85°, 1° bins
        curve = np.full(len(bins) - 1, np.nan)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        for j in range(len(bins) - 1):
            in_bin = mask & (inc_angle >= bins[j]) & (inc_angle < bins[j + 1])
            if in_bin.sum() >= 5:
                curve[j] = np.median(bs_raw[in_bin])

        # smooth with moving average
        valid_curve = np.isfinite(curve)

        if valid_curve.sum() == 0:
            continue

        if valid_curve.sum() > 3:
            curve[valid_curve] = uniform_filter1d(curve[valid_curve], size=5)

        # interpolate to get correction for each sample
        bs0 = np.interp(REFERENCE_ANGLE, bin_centers[valid_curve], curve[valid_curve])
        correction = np.interp(
            inc_angle[mask], bin_centers[valid_curve], curve[valid_curve]
        )
        bs_corr[mask] = bs_raw[mask] - correction + bs0

    bs_corr[inc_angle < nadir_cutoff] = np.nan
    bs_raw[inc_angle < nadir_cutoff] = np.nan

    valid_corr = bs_corr[np.isfinite(bs_corr)]
    if len(valid_corr) > 0:
        q01, q99 = np.percentile(valid_corr, [1, 99])
        bs_corr = np.where(np.isfinite(bs_corr), np.clip(bs_corr, q01, q99), np.nan)

    return bs_corr, bs_raw, labels


def detect_first_return(amps, pix_m, min_range_m=3.0, threshold_ratio=0.1):
    """
    Detect first bottom return using median-filtered envelope.
    Returns slant range (m) or None.
    """
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
