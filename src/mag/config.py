"""
Magnetometer algorithm constants.

These are tuning parameters for the IGRF correction, along-track
filtering, and grid interpolation. Site-independent: keep here, not
in yaml.

Reference window sizes assume ~3 m record spacing (SeaSPY2 at typical
survey speed). Adjust if survey conditions differ significantly.
"""

# ──────────────────────────────────────────────────────────
# Quality / outlier filtering
# ──────────────────────────────────────────────────────────
QUALITY_MIN = 90              # SeaSPY2 quality flag minimum
F_TOLERANCE = 0.10            # accept F within ±10% of IGRF
BG_OUTLIER_SIGMA = 2.0        # remove background points beyond this σ

# ──────────────────────────────────────────────────────────
# Along-track median filter windows (samples)
# WIN_BG  ≈ 100 m  removes local anomalies, keeps regional background
# WIN_ANOM ≈ 15 m  removes electronic noise, preserves UCH anomalies
# ──────────────────────────────────────────────────────────
WIN_BG = 35
WIN_ANOM = 5

# ──────────────────────────────────────────────────────────
# IDW interpolation
# ──────────────────────────────────────────────────────────
MAX_EXTRAP_M = 50.0           # max distance from track to interpolate
IDW_POWER = 2
IDW_NEIGHBORS = 12
IDW_CHUNK = 50_000

# ──────────────────────────────────────────────────────────
# Background spatial smoothing
# ──────────────────────────────────────────────────────────
SMOOTH_M = 200.0              # Gaussian sigma in metres
SMOOTH_WEIGHT_THRESHOLD = 0.1

# ──────────────────────────────────────────────────────────
# Target detection (post-processing on anomaly grid)
# ──────────────────────────────────────────────────────────
TARGET_THRESHOLD_NT = 100.0       # min |anomaly| to flag as target
TARGET_PEAK_FOOTPRINT_PX = 20     # local max footprint, ~10 m at 0.5 m grid
TARGET_DEDUP_RADIUS_M = 15.0      # merge dipole pairs / peak shoulders