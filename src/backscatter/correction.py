# src/backscatter/correction.py
import warnings
import numpy as np
from numpy.exceptions import RankWarning
from scipy.signal import medfilt
from sklearn.cluster import KMeans


def _build_features(bs_db, inc_angle):
    bins = np.arange(0, 91, 1)
    z    = np.zeros_like(bs_db)
    for i in range(len(bins) - 1):
        mask = (inc_angle >= bins[i]) & (inc_angle < bins[i + 1])
        if mask.sum() < 2:
            continue
        mu, sigma = bs_db[mask].mean(), bs_db[mask].std()
        if sigma > 0:
            z[mask] = (bs_db[mask] - mu) / sigma
    return np.column_stack([z, inc_angle / 90.0]).astype(np.float32)


def collect_features(bs_linear, inc_angle, sample_ratio=0.01):
    bs_db = 10 * np.log10(np.maximum(bs_linear, 1e-12))
    idx   = np.random.choice(len(bs_db),
                             max(1, int(len(bs_db) * sample_ratio)),
                             replace=False)
    return _build_features(bs_db[idx], inc_angle[idx])


def fit_kmeans(feature_list, n_clusters=7):
    all_feat = np.concatenate(feature_list, axis=0)
    km = KMeans(n_clusters=n_clusters, init="k-means++",
                n_init=10, random_state=0)
    km.fit(all_feat)
    return km


def _polyfit_with_rejection(x, y, deg, n_iter=2, sigma_thresh=2.0):
    """
    Iterative polynomial fitting with outlier rejection.
    Fit once, remove residuals > sigma_thresh * std, refit.
    """
    mask = np.ones(len(x), dtype=bool)
    coeffs = None

    for _ in range(n_iter + 1):
        if mask.sum() < deg + 2:
            break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RankWarning)
            coeffs = np.polyfit(x[mask], y[mask], deg)
        if _ < n_iter:
            residual = y - np.polyval(coeffs, x)
            std = np.std(residual[mask])
            if std > 0:
                mask = np.abs(residual) < sigma_thresh * std

    return coeffs


def angle_correction(bs_linear, inc_angle, km, n_clusters=7,
                     nadir_cutoff=10.0, poly_deg=3):
    # zero amplitude filtering
    valid_amp = bs_linear > 0
    bs_db = np.full(len(bs_linear), np.nan, dtype=np.float64)
    bs_db[valid_amp] = 10 * np.log10(bs_linear[valid_amp].astype(np.float64))

    feat   = _build_features(bs_db, inc_angle)
    labels = km.predict(feat).astype(np.int8)
    bs_corr = bs_db.copy()

    for c in range(n_clusters):
        mask = (labels == c) & (inc_angle >= nadir_cutoff) & np.isfinite(bs_db)
        if mask.sum() < poly_deg + 2:
            continue

        # iterative polyfit with outlier rejection
        coeffs = _polyfit_with_rejection(
            inc_angle[mask], bs_db[mask], poly_deg)

        if coeffs is None:
            continue

        ref_angle = np.median(inc_angle[mask])
        bs0 = np.polyval(coeffs, ref_angle)
        bs_corr[mask] = bs_db[mask] - np.polyval(coeffs, inc_angle[mask]) + bs0

    bs_corr[inc_angle < nadir_cutoff] = np.nan

    return bs_corr.astype(np.float32), labels


def detect_first_return(amps, pix_m, min_range_m=3.0, threshold_ratio=0.1):
    """
    Liu & Ye (2023): detect first bottom return.
    Applies median filter (k=21) before peak detection to suppress
    noise spikes and nadir direct-arrival artifacts.
    Returns slant range [m] of first significant return, or None.
    """
    min_idx = int(min_range_m / pix_m)
    if min_idx >= len(amps):
        return None

    # median filter suppresses noise and direct-arrival spike at nadir
    search = medfilt(amps[min_idx:].astype(np.float64), kernel_size=21)

    max_val = search.max()
    if max_val <= 0:
        return None

    threshold  = max_val * threshold_ratio
    diff       = np.diff(search)
    candidates = np.where(diff > threshold)[0]

    if len(candidates) == 0:
        return None

    return float((candidates[0] + min_idx) * pix_m)