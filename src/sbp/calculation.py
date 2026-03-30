# src/sbp/calculation.py
import numpy as np
from typing import Optional
from src.config import SOUND_SPEED

SAMPLE_DEPTH = 20480e-9 * SOUND_SPEED / 2  # 0.015247 m/sample

Z_WATER = 1000 * SOUND_SPEED # Pa·s/m

SEDIMENT_THRESHOLDS = [
    (7.33,  "Coarse sand"),                 # 0: RL < 7.33
    (8.02,  "Fine sand"),                   # 1: RL < 8.02
    (8.73,  "Very fine sand"),              # 2: RL < 8.73
    (9.63,  "Silty sand"),                  # 3: RL < 9.63
    (9.82,  "Sandy silt"),                  # 4: RL < 9.82
    (10.25, "Silt"),                        # 5: RL < 10.25
    (11.98, "Sandy-silt-clay"),             # 6: RL < 11.98
    (13.20, "Silty clay"),
    (13.37, "Clayey silt"),    # 7: RL < 13.37
    (22.40, "Framework-supported mud"),     # 8: RL < 22.40 (過渡態壓實軟泥, 孔隙率 < 88%)
    (float('inf'), "Fluid mud")             # 9: RL >= 22.40 (懸浮態流體泥, 孔隙率 >= 88%)
]

SEDIMENT_LABELS = [label for _, label in SEDIMENT_THRESHOLDS]

RL_MIN = 3.0
RL_MAX = 30.0


def classify_sediment(rl_db: float) -> int:
    """
    依據物理聲學推導之絕對閾值進行沉積物分類。
    """
    if np.isnan(rl_db) or rl_db < 0:
        return -1

    for i, (thresh, _) in enumerate(SEDIMENT_THRESHOLDS):
        if rl_db < thresh:
            return i

    return len(SEDIMENT_THRESHOLDS) - 1


def rl_to_impedance(rl_db: float | np.ndarray) -> float | np.ndarray:
    R = np.power(10.0, -np.asarray(rl_db) / 20.0)
    R = np.clip(R, 0.0, 0.9999)
    return Z_WATER * (1.0 + R) / (1.0 - R)


def estimate_cc(amps: np.ndarray,
                blanking: int = 50,
                tight_win: int = 50) -> Optional[float]:
    """
    Estimate calibration coefficient CC from a single ping.
    CC = (r1 * A1)^2 / (r2 * A2)
    where r1*A1 = range*amplitude of bottom echo (B),
          r2*A2 = range*amplitude of bottom-surface-bottom echo (BSB).
    Ref: Huang & Liu (2015) Eq. (6)
    """
    if len(amps) < 500:
        return None
    idx_b = int(np.argmax(amps[blanking:])) + blanking
    amp_b = float(amps[idx_b])
    if amp_b <= 0:
        return None
    center    = idx_b * 2
    bsb_start = max(center - tight_win, int(idx_b * 1.9))
    bsb_end   = min(len(amps), center + tight_win)
    if bsb_end <= bsb_start:
        return None
    idx_bsb = int(np.argmax(amps[bsb_start:bsb_end])) + bsb_start
    amp_bsb = float(amps[idx_bsb])
    if amp_bsb <= 0 or amp_bsb < amp_b * 0.05:
        return None
    return (idx_b * amp_b) ** 2 / (idx_bsb * amp_bsb)


def compute_rl(amps, cc, blanking=50):
    """
    Compute reflection loss from a single ping.
    RL = -20 * log10(r1 * A1 / CC)
    Ref: Huang & Liu (2015) Eq. (9)
    """
    if len(amps) < 100 or cc <= 0:
        return None
    search = amps[blanking:]
    if len(search) == 0:
        return None
    idx_b = int(np.argmax(search)) + blanking
    amp_b = float(amps[idx_b])
    if amp_b <= 0:
        return None
    R = (idx_b * amp_b) / cc
    if R <= 0:
        return None
    rl = -20.0 * np.log10(R)
    return float(np.clip(rl, RL_MIN, RL_MAX))


def compute_global_cc(jsf_paths, read_sbp_jsf_fn, vrm_tif, bs_tif,
                      transformer, sample_raster_fn,
                      min_consec: int = 20) -> float:
    """
    Compute global calibration coefficient from flat, uniform-substrate segments.
    Uses VRM (low = flat) and SSS backscatter (low std = uniform substrate)
    to select high-quality calibration segments.
    """
    jsf_list  = list(jsf_paths)
    all_vrm   = []

    for jsf in jsf_list:
        data = read_sbp_jsf_fn(jsf)
        if 'SBP' not in data:
            continue
        d     = data['SBP']
        valid = ~np.isnan(d['lon'])
        if valid.sum() == 0:
            continue
        x, y  = transformer.transform(d['lon'][valid], d['lat'][valid])
        vrm   = sample_raster_fn(vrm_tif, x, y)
        all_vrm.extend(vrm[np.isfinite(vrm)].tolist())

    if not all_vrm:
        raise RuntimeError("No valid VRM data found.")

    vrm_thresh = float(np.percentile(all_vrm, 25))
    candidates = []

    for jsf in jsf_list:
        data = read_sbp_jsf_fn(jsf)
        if 'SBP' not in data:
            continue
        d     = data['SBP']
        valid = ~np.isnan(d['lon'])
        if valid.sum() < min_consec:
            continue
        x, y  = transformer.transform(d['lon'][valid], d['lat'][valid])
        orig  = np.where(valid)[0]
        vrm   = sample_raster_fn(vrm_tif, x, y)
        bs    = sample_raster_fn(bs_tif, x, y)
        quality = np.isfinite(vrm) & (vrm < vrm_thresh) & np.isfinite(bs)

        start, end = _find_longest_run(quality)
        if end - start < min_consec:
            continue
        bs_seg_std = float(np.nanstd(bs[start:end]))
        candidates.append({
            'jsf': jsf, 'valid': valid, 'orig': orig,
            'start': start, 'end': end, 'bs_std': bs_seg_std,
        })

    if not candidates:
        raise RuntimeError("No flat segments found for calibration.")

    bs_stds       = np.array([c['bs_std'] for c in candidates])
    bs_std_thresh = float(np.percentile(bs_stds, 25))
    good          = [c for c in candidates if c['bs_std'] <= bs_std_thresh]
    all_cc        = []

    for c in good:
        data       = read_sbp_jsf_fn(c['jsf'])['SBP']
        amps_valid = data['amps'][c['valid']]
        for i in range(c['start'], c['end']):
            cc = estimate_cc(amps_valid[i])
            if cc is not None and cc > 0:
                all_cc.append(cc)

    if not all_cc:
        raise RuntimeError("No valid CC computed.")

    all_cc = np.array(all_cc)
    q25, q75 = np.percentile(all_cc, 25), np.percentile(all_cc, 75)
    iqr      = q75 - q25
    mask     = (all_cc >= q25 - 1.5 * iqr) & (all_cc <= q75 + 1.5 * iqr)
    return float(np.median(all_cc[mask]))


def _find_longest_run(bool_arr):
    best_s, best_n = 0, 0
    cur_s,  cur_n  = 0, 0
    for i, v in enumerate(bool_arr):
        if v:
            cur_n += 1
            if cur_n > best_n:
                best_n, best_s = cur_n, cur_s
        else:
            cur_s, cur_n = i + 1, 0
    return best_s, best_s + best_n