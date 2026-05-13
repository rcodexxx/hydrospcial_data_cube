"""
Reflection Loss (RL) computation and sediment classification.

RL thresholds are derived analytically from sediment density and Vp/Vw
ratios using freshwater acoustics. This avoids using Hamilton (1980)'s
seawater-calibrated RL values directly, which would be biased for
freshwater environments.

References:
  Hamilton (1980) — continental terrace sediment parameters (Table IB)
  Holland (2002) — Sicily AoI inversion, partially consolidated mud
  McAnally et al. (2007) — fluid mud
  Huang (2015) — CC calibration and RL formula
"""
from typing import Optional

import numpy as np

from src.config import get_config
from src.sbp.config import (
    BLANKING_SAMPLES, RL_MIN, RL_MAX,
    CC_TIGHT_WIN_SAMPLES, CC_MIN_BSB_RATIO,
)


# ──────────────────────────────────────────────────────────
# Environment-dependent constants (resolved from yaml at first call)
# ──────────────────────────────────────────────────────────
def _sound_speed():
    return get_config()["environment"]["sound_speed"]


def _z_water():
    return 1000.0 * _sound_speed()   # freshwater: ρ = 1000 kg/m³


def _sample_depth():
    # 20480 ns / 2 (one-way) × speed
    return 20480e-9 * _sound_speed() / 2.0


# ──────────────────────────────────────────────────────────
# Sediment physical properties (density g/cc, Vp/Vw ratio)
# ──────────────────────────────────────────────────────────
SEDIMENT_PROPERTIES = [
    # (name, density_g_cc, vp_ratio)
    ("Sand",                     1.943, 1.184),   
    ("Silty sand / Sandy silt",  1.772, 1.107),   
    ("Silt",                     1.740, 1.085),   
    ("Sand-silt-clay",           1.596, 1.060),   
    ("Clayey silt / Silty clay", 1.455, 1.030),   
    ("Partially consolidated mud", 1.320, 0.989), 
    ("Fluid mud",                1.100, 0.974),   
]


def _compute_rl_thresholds():
    """
    Derive per-sediment RL values and classification thresholds.
    Called lazily (when SEDIMENT_THRESHOLDS first accessed).
    """
    c = _sound_speed()
    z_w = _z_water()

    results = []
    for name, rho_g_cc, vp_ratio in SEDIMENT_PROPERTIES:
        rho = rho_g_cc * 1000.0
        vp = vp_ratio * c
        z_sed = rho * vp
        r = abs((z_sed - z_w) / (z_sed + z_w))
        rl_db = -20.0 * np.log10(max(r, 1e-10))
        results.append((name, rho, vp, z_sed, rl_db))

    # Sort by RL ascending (strong reflector → weak reflector)
    results.sort(key=lambda x: x[4])

    # Classification thresholds: midpoint between adjacent sediment types
    thresholds = []
    for i, (name, rho, vp, z, rl) in enumerate(results):
        if i < len(results) - 1:
            rl_next = results[i + 1][4]
            t = (rl + rl_next) / 2.0
        else:
            t = float("inf")
        thresholds.append((t, name))
    return results, thresholds


# Cached on first call (avoids recomputing per RL lookup)
_cached = {"data": None, "thresholds": None, "labels": None}


def _ensure_cached():
    if _cached["data"] is None:
        data, thresholds = _compute_rl_thresholds()
        _cached["data"] = data
        _cached["thresholds"] = thresholds
        _cached["labels"] = [name for _, name in thresholds]


def get_sediment_labels():
    _ensure_cached()
    return _cached["labels"]


def get_sediment_thresholds():
    _ensure_cached()
    return _cached["thresholds"]


def print_threshold_table():
    """Diagnostic: print derived RL thresholds and sediment parameters."""
    _ensure_cached()
    print(f"{'Sediment':<22s} {'ρ kg/m³':>9s} {'Vp m/s':>8s} "
          f"{'Z_sed':>10s} {'RL dB':>7s} {'Upper':>8s}")
    print("-" * 72)
    for (name, rho, vp, z, rl), (t, _) in zip(
        _cached["data"], _cached["thresholds"]
    ):
        t_str = f"{t:.2f}" if t != float("inf") else "∞"
        print(f"{name:<22s} {rho:9.0f} {vp:8.1f} {z:10.3e} "
              f"{rl:7.2f} {t_str:>8s}")


# ──────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────
def classify_sediment(rl_db: float) -> int:
    if np.isnan(rl_db) or rl_db < 0:
        return -1
    _ensure_cached()
    for i, (thresh, _) in enumerate(_cached["thresholds"]):
        if rl_db < thresh:
            return i
    return len(_cached["thresholds"]) - 1


def rl_to_vp(rl_db: float) -> float:
    """
    Map RL to Vp by classifying first, then returning that sediment's Vp.
    Used by isopach depth conversion.
    """
    if not np.isfinite(rl_db) or rl_db < 0:
        return _sound_speed()
    idx = classify_sediment(rl_db)
    if idx < 0:
        return _sound_speed()
    _ensure_cached()
    return float(_cached["data"][idx][2])


def find_bsb_in_average(avg_trace: np.ndarray, idx_b: int) -> Optional[int]:
    """
    Locate BSB peak in a segment-averaged trace.

    Searches within 2 × idx_B ± 7.5% (the BSB window from
    Huang & Liu 2015). Used by both estimate_cc and method figures
    to ensure consistency.

    Returns
    -------
    idx_bsb : int or None
        Sample index of BSB peak, or None if window is too narrow.
    """
    bsb_start = int(idx_b * 1.95)
    bsb_end = min(len(avg_trace), int(idx_b * 2.10))
    if bsb_end - bsb_start < 10:
        return None
    return int(np.argmax(avg_trace[bsb_start:bsb_end])) + bsb_start


def estimate_cc(amps_segment: np.ndarray) -> Optional[float]:
    """
    Estimate CC from a stable calibration segment via ping averaging.
    See Huang & Liu 2015, Eq. (6).
    """
    if amps_segment.ndim != 2 or amps_segment.shape[0] < 30:
        return None

    avg = np.mean(amps_segment, axis=0)
    if len(avg) < 100:
        return None

    from src.sbp.sbr_picking import find_bottom
    idx_b = find_bottom(avg)
    amp_b = float(avg[idx_b])
    if amp_b <= 0:
        return None

    idx_bsb = find_bsb_in_average(avg, idx_b)
    if idx_bsb is None:
        return None

    amp_bsb = float(avg[idx_bsb])
    if amp_bsb <= 0 or amp_bsb >= amp_b:
        return None

    return amp_b ** 2 / amp_bsb


def compute_rl_batch(amps_2d: np.ndarray, cc: float) -> np.ndarray:
    n_pings = amps_2d.shape[0]
    rl = np.full(n_pings, np.nan, dtype=np.float64)
    if cc <= 0 or amps_2d.shape[1] < 100:
        return rl

    search = amps_2d[:, BLANKING_SAMPLES:]
    idx_b = np.argmax(search, axis=1) + BLANKING_SAMPLES
    amp_b = amps_2d[np.arange(n_pings), idx_b]

    valid = amp_b > 0
    ratio = np.where(valid, amp_b / cc, 0.0)
    valid &= ratio > 0

    rl[valid] = np.clip(-20.0 * np.log10(ratio[valid]), RL_MIN, RL_MAX)
    return rl


if __name__ == "__main__":
    import os
    os.environ.setdefault("HYDRO_CONFIG", "configs/mudan.yaml")
    print_threshold_table()