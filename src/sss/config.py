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
NADIR_CUTOFF_DEG = 15.0      # samples below this angle are discarded
FAR_CUTOFF_HF = 76.0
FAR_CUTOFF_LF = 70.0


# ── Angular waterfall binning ────────────────────────────────
ANGLE_MIN_DEG      = 20.0
ANGLE_MAX_DEG      = 70.0
ANGLE_BIN_WIDTH_DEG = 1.0

# ── Ping rejection (turn filter) ─────────────────────────────
TURN_HEADING_THRESHOLD_DEG    = 3.0   # per-ping heading delta
ROLL_THRESHOLD_DEG            = 3
HEADING_RATE_THRESHOLD_DEG_S  = 2

# ── Statistical thresholds ───────────────────────────────────
MIN_SAMPLES_PER_ANGLE_BIN   = 10  # for ABC curve estimation
MIN_SAMPLES_PER_CLUSTER_BIN = 5   # for per-cluster ABC

# ── Altitude outlier filter ──────────────────────────────────
ALTITUDE_MEDIAN_FILTER_WINDOW = 11   # pings
ALTITUDE_OUTLIER_RATIO        = 0.2  # drop if |alt - local_median| / local_median > this

# ── Cluster post-processing ──────────────────────────────────
MODE_FILTER_WINDOW = 3   # pixels

# ── Mosaic weighting ─────────────────────────────────────────
MOSAIC_WEIGHT_FLOOR        = 0.05
MOSAIC_FAR_ANGLE_PENALTY   = 0.3
IDW_SEARCH_RADIUS_M        = 5
IDW_NEIGHBORS              = 8


def get_far_cutoff(channel_or_freq):
    s = str(channel_or_freq).upper()
    if "HF" in s:
        return FAR_CUTOFF_HF
    if "LF" in s:
        return FAR_CUTOFF_LF
    raise ValueError(f"cannot determine frequency from {channel_or_freq!r}")


def infer_frequency(channels):
    has_hf = any("HF" in c.upper() for c in channels)
    has_lf = any("LF" in c.upper() for c in channels)
    if has_hf and has_lf:
        raise ValueError(f"mixed-frequency channels not supported: {channels}")
    if has_hf:
        return "HF"
    if has_lf:
        return "LF"
    raise ValueError(f"no recognizable frequency in {channels}")