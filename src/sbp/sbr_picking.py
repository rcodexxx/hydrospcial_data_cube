"""
Sub-Bottom Reflector (SBR) detection.

Two pickers:
  - pick_sbr_thickness: lenient (used for diagnostic plots only)
  - pick_sbr_thickness_strict: CFAR-based strict picker for hybrid isopach.

Strict picker uses Constant False Alarm Rate (CFAR) thresholding
(Schock 2004 IEEE J. Oceanic Eng.): peaks in the detrended envelope
must rise N × MAD-derived noise above the local noise floor. The
threshold is adaptive per-ping, automatically handling spatially
varying noise across the dataset.
"""
import numpy as np
from scipy.signal import find_peaks, hilbert

from src.sbp.config import (
    BLANKING_SAMPLES,
    SBR_BLIND_ZONE_M, SBR_MAX_DEPTH_M,
    SONAR_DRAFT_M, SBR_MULTIPLE_REJECT_M,
)

NS_PER_SAMPLE = 20480e-9

# CFAR detection parameters
CFAR_N_SIGMA = 5.0           # detection threshold (5σ ~ 1e-7 false alarm rate)
MAD_TO_SIGMA = 1.4826        # MAD → σ conversion (Gaussian assumption)
STRICT_DOMINANCE_DB = 3.0    # top peak must beat second-best by this much
STRICT_MIN_WIDTH = 3         # samples


def find_bottom(amps_1d):
    """Locate primary seafloor return (max amplitude after blanking)."""
    return int(np.argmax(amps_1d[BLANKING_SAMPLES:])) + BLANKING_SAMPLES


def pick_sbr_thickness(amps, idx_b, local_vp):
    """Lenient picker (diagnostic use only)."""
    local_sdep = NS_PER_SAMPLE * local_vp / 2.0
    s = idx_b + int(SBR_BLIND_ZONE_M / local_sdep)
    e = min(len(amps), idx_b + int(SBR_MAX_DEPTH_M / local_sdep))
    if e - s < 10:
        return None

    envelope = np.abs(hilbert(amps[s:e].astype(np.float64)))
    db_win = 20 * np.log10(np.maximum(envelope, 1.0))
    noise_floor = np.median(db_win) + 5.0

    peaks, properties = find_peaks(
        db_win, prominence=3.0,
        distance=int(SBR_BLIND_ZONE_M / local_sdep),
        height=noise_floor, width=2,
    )
    if len(peaks) == 0:
        return None
    best = np.argmax(properties["prominences"] * properties["peak_heights"])
    thick = (s + peaks[best] - idx_b) * local_sdep
    if abs(thick - SONAR_DRAFT_M) < SBR_MULTIPLE_REJECT_M:
        return None
    return thick


def pick_sbr_thickness_strict(amps, idx_b, local_vp):
    """
    CFAR-based strict SBR picker.

    Returns
    -------
    dict | None
        {'thickness_m', 'snr', 'prominence_db'} if a peak passes
        CFAR + dominance + width tests; None otherwise.
    """
    local_sdep = NS_PER_SAMPLE * local_vp / 2.0
    s = idx_b + int(SBR_BLIND_ZONE_M / local_sdep)
    e = min(len(amps), idx_b + int(SBR_MAX_DEPTH_M / local_sdep))
    if e - s < 10:
        return None

    envelope = np.abs(hilbert(amps[s:e].astype(np.float64)))
    db_win = 20 * np.log10(np.maximum(envelope, 1.0))

    # Detrend ringdown decay
    x = np.arange(len(db_win))
    coef = np.polyfit(x, db_win, 1)
    detrended = db_win - (coef[0] * x + coef[1])

    # CFAR threshold from MAD-derived noise σ
    noise_med = np.median(detrended)
    mad = np.median(np.abs(detrended - noise_med))
    sigma = MAD_TO_SIGMA * mad
    if sigma < 1e-6:
        return None  # degenerate envelope (constant)
    cfar_prominence = CFAR_N_SIGMA * sigma

    peaks, properties = find_peaks(
        detrended,
        prominence=cfar_prominence,
        distance=int(SBR_BLIND_ZONE_M / local_sdep),
        width=STRICT_MIN_WIDTH,
    )
    if len(peaks) == 0:
        return None

    proms = properties["prominences"]
    sorted_p = np.sort(proms)[::-1]

    # Dominance: top peak must clearly beat second-best
    if len(sorted_p) >= 2 and sorted_p[0] - sorted_p[1] < STRICT_DOMINANCE_DB:
        return None

    best = np.argmax(proms)
    thick = (s + peaks[best] - idx_b) * local_sdep
    if abs(thick - SONAR_DRAFT_M) < SBR_MULTIPLE_REJECT_M:
        return None

    return {
        'thickness_m': thick,
        'snr': proms[best] / sigma,
        'prominence_db': proms[best],
    }