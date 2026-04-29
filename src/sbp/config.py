"""
SBP algorithm constants.

These are tuning parameters for the radiometric correction and SBR picking.
Keep these here rather than in yaml: they depend on the method, not the site.
"""

# ──────────────────────────────────────────────────────────
# RL computation
# ──────────────────────────────────────────────────────────
BLANKING_SAMPLES = 50         # skip near-transducer samples
RL_MIN = 0.0                   # physical min (strong reflector limit)
RL_MAX = 50.0                  # physical max (weak reflector / fluid mud)

# ──────────────────────────────────────────────────────────
# CC estimation (Huang & Liu 2015)
# ──────────────────────────────────────────────────────────
CC_TIGHT_WIN_SAMPLES = 50      # BSB search window around 2×idx_B
CC_MIN_BSB_RATIO = 0.05        # BSB echo must be at least 5% of B amplitude

# ──────────────────────────────────────────────────────────
# CC calibration — segment selection criteria
#
# Physical reasoning for each parameter:
#   - VRM/slope/BPI: multi-scale terrain flatness (smooth-interface
#     condition required by Huang & Liu 2015)
#   - depth_std:     within-segment depth uniformity
#   - amp_CV:        within-segment substrate uniformity (the BSB method
#                    assumes B and BSB reflect off the same homogeneous
#                    substrate; high CV violates this assumption)
#   - min_start_ping: skip warm-up at the beginning of each survey line
#                    (gain stabilisation, towfish attitude)
# ──────────────────────────────────────────────────────────
CC_MIN_CONSEC_PINGS = 100       # min consecutive good pings per segment
CC_VRM_PERCENTILE = 20           # VRM threshold percentile
CC_SLOPE_MAX_DEG = 6.0           # max local slope (degrees)
CC_BPI_ABS_MAX = 0.3             # max abs(BPI) (m, ±0.3 m around mean)
CC_DEPTH_STD_MAX = 0.5           # max depth std within segment (m)
CC_AMP_CV_MAX = 0.35             # max amp_b coefficient of variation in segment
CC_MIN_START_PING = 50           # skip first N pings of each file (warm-up)

# ──────────────────────────────────────────────────────────
# RL post-processing
# ──────────────────────────────────────────────────────────
RL_SPIKE_WINDOW = 21             # median filter window for spike detection
RL_SPIKE_THRESHOLD = 5.0         # dB, deviation from local median
RL_SMOOTH_WINDOW = 11            # along-track smoothing window

# ──────────────────────────────────────────────────────────
# Sediment-feature regression (build_sediment.py)
# ──────────────────────────────────────────────────────────
RF_MAX_DIST_M = 100.0            # interpolate only within this range of tracks
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 5

# ──────────────────────────────────────────────────────────
# Isopach (SBR picking)
# ──────────────────────────────────────────────────────────
SBR_BLIND_ZONE_M = 0.15          # skip this much below bottom (saturation)
SBR_MAX_DEPTH_M = 4.0            # max sediment depth to search
SBR_PROMINENCE_DB = 3.0           # peak prominence threshold
SBR_MIN_WIDTH = 2                # min peak width in samples
SONAR_DRAFT_M = 1.04             # transducer draft (assumed fixed)
SBR_MULTIPLE_REJECT_M = 0.15     # reject picks within this range of sonar draft

# Thickness bounds
THICK_MIN_M = 0.1                # below blind zone, won't trigger
THICK_MAX_M = 3.0                # reservoir-specific upper bound

# ──────────────────────────────────────────────────────────
# IDW interpolation (isopach)
# ──────────────────────────────────────────────────────────
IDW_MAX_GAP_M = 70.0             # max gap between tracklines for interp
IDW_K_NEIGHBORS = 12              # typical value (was 200, too many)
IDW_EPS = 1e-6                    # regularization to avoid /0

# ──────────────────────────────────────────────────────────
# Sediment classification color palette (7 classes after merging).
# Single source of truth, used by:
#   - scripts/plot/plot_sub_bottom.py (thesis figures)
#   - scripts/build/build_sediment_rgb.py (viewer GeoTIFF)
#   - api_server SEDIMENT_LABELS for UI legend
# ──────────────────────────────────────────────────────────
SEDIMENT_COLORS = [
    "#A0522D",  # 0: Coarse sand
    "#CD853F",  # 1: Fine sand / Silty sand
    "#DEB887",  # 2: Silt / Sandy silt
    "#BDB76B",  # 3: Sand-silt-clay
    "#8FBC8F",  # 4: Compacted mud
    "#6495ED",  # 5: Clayey silt / Silty clay
    "#2E5C8A",  # 6: Fluid mud
]

SEDIMENT_LABELS = [
    "Coarse sand",
    "Fine sand / Silty sand",
    "Silt / Sandy silt",
    "Sand-silt-clay",
    "Compacted mud",
    "Clayey silt / Silty clay",
    "Fluid mud",
]