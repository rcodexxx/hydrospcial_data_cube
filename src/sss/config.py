"""
SSS algorithm constants.

These are NOT user-facing config. They encode physical assumptions,
statistical stability thresholds, and algorithm internals. Changing
them requires re-validating results.

For site-specific settings, see configs/*.yaml.
"""

# ── Reference angles ─────────────────────────────────────────
REFERENCE_ANGLE_DEG = 45.0   # Zhao (2017) reference for BS_0

# ── Angle masking ────────────────────────────────────────────
# Frequency-specific cutoffs from angle_snr diagnostic (Mudan reservoir).
# HF 65°: signal-character plateau onset (conservative).
# LF 70°: signal-failure knee (CV jumps, median dB drops).
NADIR_CUTOFF_DEG = 15.0
FAR_CUTOFF_HF = 65.0
FAR_CUTOFF_LF = 70.0

# ── Angular waterfall binning ────────────────────────────────
ANGLE_MIN_DEG = 20.0
ANGLE_MAX_DEG = 70.0
ANGLE_BIN_WIDTH_DEG = 1.0

# ── Ping rejection (turn filter) ─────────────────────────────
TURN_HEADING_THRESHOLD_DEG = 3.0
ROLL_THRESHOLD_DEG = 3
HEADING_RATE_THRESHOLD_DEG_S = 2

# ── Statistical thresholds ───────────────────────────────────
MIN_SAMPLES_PER_ANGLE_BIN   = 10
MIN_SAMPLES_PER_CLUSTER_BIN = 5

# ── Cluster post-processing ──────────────────────────────────
MODE_FILTER_WINDOW = 3

# ── Mosaic weighting ─────────────────────────────────────────
MOSAIC_WEIGHT_FLOOR = 0.05
MOSAIC_FAR_ANGLE_PENALTY = 0.3
IDW_SEARCH_RADIUS_M = 1.5
IDW_NEIGHBORS = 4

# ── Waterfall rendering (slant→ground correction) ────────────
MAX_GROUND_RANGE_M = 50.0
GROUND_RANGE_RESOLUTION_M = 0.1

# ── Kalman bottom-tracking smoothing (Woock 2011) ────────────
# Random-walk altitude state model: state = [altitude]
# Q: process noise variance — how much altitude wanders ping-to-ping
# R: measurement noise variance — typical first-return detection error
# σ-gate: reject obs if Mahalanobis distance > this many std devs
KALMAN_PROCESS_VAR = 0.01
KALMAN_MEASUREMENT_VAR = 9.0
KALMAN_OUTLIER_SIGMA = 3.0

# ── Gain normalization (Stage 2b) ────────────────────────────
GAIN_NORM_PERCENTILE = 40           # ref percentile per (line, channel)
GAIN_NORM_MIN_SAMPLES = 100         # skip groups smaller than this

# ── Confidence (angular Gaussian) ────────────────────────────
CONFIDENCE_ANGULAR_SIGMA_DEG = 12.0


def get_far_cutoff(channel_or_freq):
    """Resolve far cutoff angle by channel name or frequency string."""
    s = str(channel_or_freq).upper()
    if "HF" in s:
        return FAR_CUTOFF_HF
    if "LF" in s:
        return FAR_CUTOFF_LF
    raise ValueError(f"cannot determine frequency from {channel_or_freq!r}")


def infer_frequency(channels):
    """Determine 'HF' or 'LF' from a channel list. Mixed lists not allowed."""
    has_hf = any("HF" in c.upper() for c in channels)
    has_lf = any("LF" in c.upper() for c in channels)
    if has_hf and has_lf:
        raise ValueError(f"mixed-frequency channels not supported: {channels}")
    if has_hf:
        return "HF"
    if has_lf:
        return "LF"
    raise ValueError(f"no recognizable frequency in {channels}")


def channels_for_freq(freq):
    """Return canonical 4-channel list for a given frequency."""
    f = freq.upper()
    if f not in ("HF", "LF"):
        raise ValueError(f"unknown frequency: {freq!r}")
    return [f"{f}_port", f"{f}_stbd"]