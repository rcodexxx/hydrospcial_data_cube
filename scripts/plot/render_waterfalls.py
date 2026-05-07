"""
Batch render SSS waterfall and SBP envelope images.

SSS:
  - Per-ping bottom tracking via threshold first-return detection
  - Kalman smoothing on detected altitude sequence (Woock 2011)
    with Mahalanobis gating to reject first-return failures
  - Slant→ground range correction per ping with channel-aware
    angle cutoff (HF 65°, LF 70°)
  - Water column samples removed (slant < altitude / cos(15°))
  - Per-channel AGC normalization (40th percentile reference,
    industry-standard single-file gain matching)
  - Resampled to a uniform 0–WATERFALL_MAX_GROUND_M ground-range grid
  - Inner cropping (np.all): drops every column that has NaN in any
    ping, leaving aligned inner edges across all pings in the file
  - Port (flipped) + fixed-width connector + stbd merged. Connector
    is a 2m visual strip (NADIR_CONNECTOR_M) marking the nadir region;
    no real data lives there
  - Cyan nadir marker line drawn at the centre of the connector
  - Display in dB (matches mosaic output unit)

SBP: envelope profile (x=along-track, y=depth) — unchanged.

Output:
  outputs/waterfalls/sss/{filename}_HF.png
  outputs/waterfalls/sss/{filename}_LF.png
  outputs/waterfalls/sbp/{filename}.png
  outputs/waterfalls/index.json
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from tqdm import tqdm

from src.config import ROOT, get_config
from src.sbp.read_sbp_jsf import read_sbp_jsf
from src.sss.config import (
    NADIR_CUTOFF_DEG,
    GROUND_RANGE_RESOLUTION_M,
    KALMAN_PROCESS_VAR,
    KALMAN_MEASUREMENT_VAR,
    KALMAN_OUTLIER_SIGMA,
    GAIN_NORM_PERCENTILE,
    get_far_cutoff,
)
from src.sss.correction import to_db
from src.sss.georef import detect_first_return
from src.sss.read_sss_jsf import read_sss_jsf

cfg = get_config()
sss_cfg = cfg["sss"]
sbp_cfg = cfg["sbp"]

OUT_SSS = ROOT / "outputs/waterfalls/sss"
OUT_SBP = ROOT / "outputs/waterfalls/sbp"
OUT_INDEX = ROOT / "outputs/waterfalls/index.json"
OUT_SSS.mkdir(parents=True, exist_ok=True)
OUT_SBP.mkdir(parents=True, exist_ok=True)

DPI = 100
MAX_WIDTH_PX = 1400

# Display range per side. Smaller than mosaic far cutoff range
# (which extends to altitude * tan(65-70°)) — trims the noisy
# far-field part for cleaner waterfall display.
WATERFALL_MAX_GROUND_M = 45.0

# Visual connector between port and stbd halves. Not real data —
# inner cropping (np.all) removes every column that has NaN in any
# ping, so port and stbd halves end at their innermost shared data
# column. The connector inserts a fixed 2m strip between them,
# marked by a cyan nadir line at the centre. Width is a visual
# choice (~3% of total image width); narrow enough that the
# central black region looks like a deliberate marker rather than
# missing data.
NADIR_CONNECTOR_M = 2.0


def estimate_altitudes(amps_2d, pix_m_arr):
    """
    Per-ping first-return detection followed by Kalman smoothing
    with Mahalanobis gating (Woock 2011).
    """
    n_pings = amps_2d.shape[0]
    raw_alts = np.full(n_pings, np.nan, dtype=np.float32)

    for i in range(n_pings):
        pix_m = float(pix_m_arr[i]) if i < len(pix_m_arr) else np.nan
        if not np.isfinite(pix_m) or pix_m <= 0:
            continue
        alt = detect_first_return(amps_2d[i], pix_m)
        if alt is not None:
            raw_alts[i] = alt

    valid_mask = np.isfinite(raw_alts)
    if valid_mask.sum() < 5:
        return raw_alts

    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[float(np.nanmedian(raw_alts))]])
    kf.P = np.array([[10.0]])
    kf.F = np.array([[1.0]])
    kf.H = np.array([[1.0]])
    kf.Q = np.array([[KALMAN_PROCESS_VAR]])
    kf.R = np.array([[KALMAN_MEASUREMENT_VAR]])

    smoothed = np.empty(n_pings, dtype=np.float32)
    for i in range(n_pings):
        kf.predict()
        if valid_mask[i]:
            obs = raw_alts[i]
            innov = obs - kf.x[0, 0]
            innov_std = np.sqrt(kf.P[0, 0] + KALMAN_MEASUREMENT_VAR)
            if abs(innov) / max(innov_std, 1e-6) < KALMAN_OUTLIER_SIGMA:
                kf.update(np.array([[obs]]))
        smoothed[i] = float(kf.x[0, 0])

    return smoothed


def slant_to_ground_grid(amps, altitude, pix_m, ground_grid, freq):
    """
    Resample one half-ping from slant to ground range with
    channel-aware angle cutoff (NADIR ≤ angle ≤ FAR(freq)).
    Returns NaN-filled array; inner water-column region (where slant
    < min_slant) is left as NaN.
    """
    n_samples = len(amps)
    slant = np.arange(n_samples, dtype=np.float32) * pix_m

    far_cutoff = get_far_cutoff(freq)
    min_slant = altitude / np.cos(np.deg2rad(NADIR_CUTOFF_DEG))
    max_slant = altitude / np.cos(np.deg2rad(far_cutoff))
    valid = (slant >= min_slant) & (slant <= max_slant)
    if not valid.any():
        return np.full(len(ground_grid), np.nan, dtype=np.float32)

    ground = np.sqrt(np.maximum(slant[valid] ** 2 - altitude ** 2, 0.0))
    amps_v = amps[valid].astype(np.float32)
    out = np.interp(ground_grid, ground, amps_v,
                    left=np.nan, right=np.nan)
    return out.astype(np.float32)


def _agc_norm(amps):
    """Per-channel AGC: divide by 40th-percentile of positive amps."""
    positive = amps[amps > 0]
    if positive.size == 0:
        return amps.astype(np.float32)
    ref = float(np.percentile(positive, GAIN_NORM_PERCENTILE))
    return (amps.astype(np.float32) / max(ref, 1e-6)).astype(np.float32)


def _crop_water_column(port_grid, stbd_grid):
    """
    Drop inner columns that are NaN in any ping. Inner cropping is
    aggressive (np.all): we keep only columns where every ping has
    data. The trade-off is that some near-nadir samples (slant just
    above min_slant for shallow pings) are discarded — but those
    samples sit in the low-grazing-angle region with poor signal
    quality and minimal value for UCH detection. The gain is a
    clean, ragged-free inner edge that aligns across pings within
    the file.

    Both port_grid and stbd_grid use the same ground_grid layout:
    index 0 = nadir (inner), high index = outer. So inner = low
    index for both.
    """
    port_has_data = np.all(np.isfinite(port_grid), axis=0)
    if port_has_data.any():
        first_valid = int(np.where(port_has_data)[0].min())
        port_out = port_grid[:, first_valid:]
    else:
        port_out = port_grid

    stbd_has_data = np.all(np.isfinite(stbd_grid), axis=0)
    if stbd_has_data.any():
        first_valid = int(np.where(stbd_has_data)[0].min())
        stbd_out = stbd_grid[:, first_valid:]
    else:
        stbd_out = stbd_grid

    return port_out, stbd_out


def render_sss(jsf_path, ground_grid):
    """Render slant-corrected, AGC-normalized waterfall for HF and LF."""
    try:
        data = read_sss_jsf(jsf_path)
    except Exception as e:
        print(f"  Error reading {jsf_path.name}: {e}")
        return []

    results = []
    stem = jsf_path.stem
    n_ground = len(ground_grid)

    for freq in ("HF", "LF"):
        port_ch = f"{freq}_port"
        stbd_ch = f"{freq}_stbd"
        if port_ch not in data or stbd_ch not in data:
            continue

        port_cd = data[port_ch]
        stbd_cd = data[stbd_ch]
        port_amps = port_cd["amps"]
        stbd_amps = stbd_cd["amps"]

        n_pings = min(port_amps.shape[0], stbd_amps.shape[0])
        if n_pings < 5:
            continue

        port_amps = port_amps[:n_pings]
        stbd_amps = stbd_amps[:n_pings]

        pix_m_arr = port_cd["pix_m"][:n_pings]
        altitudes = estimate_altitudes(port_amps, pix_m_arr)

        port_amps_norm = _agc_norm(port_amps)
        stbd_amps_norm = _agc_norm(stbd_amps)

        port_ground = np.full((n_pings, n_ground), np.nan, dtype=np.float32)
        stbd_ground = np.full((n_pings, n_ground), np.nan, dtype=np.float32)
        for i in range(n_pings):
            alt = altitudes[i]
            pix_m = float(pix_m_arr[i]) if i < len(pix_m_arr) else 0.0
            if not (np.isfinite(alt) and pix_m > 0):
                continue
            port_ground[i] = slant_to_ground_grid(
                port_amps_norm[i], alt, pix_m, ground_grid, freq)
            stbd_ground[i] = slant_to_ground_grid(
                stbd_amps_norm[i], alt, pix_m, ground_grid, freq)

        # Drop inner all-NaN columns per side
        port_cropped, stbd_cropped = _crop_water_column(port_ground, stbd_ground)

        # Insert fixed-width connector strip (NaN, will render as
        # facecolor #101010 with cyan nadir line on top)
        gap_px = int(NADIR_CONNECTOR_M / GROUND_RANGE_RESOLUTION_M)
        gap = np.full((n_pings, gap_px), np.nan, dtype=np.float32)

        # Merge: port (flipped, outer→inner) + connector + stbd (inner→outer)
        merged = np.hstack([port_cropped[:, ::-1], gap, stbd_cropped])

        # Convert to dB for display
        merged_db = to_db(merged)
        finite = merged_db[np.isfinite(merged_db)]
        if len(finite) < 100:
            continue
        vmin, vmax = np.percentile(finite, [2, 98])

        # Auto-crop outer all-NaN columns (when this file's altitude
        # is too shallow to fill WATERFALL_MAX_GROUND_M)
        col_has_data = np.any(np.isfinite(merged_db), axis=0)
        if col_has_data.any():
            col_lo = int(np.argmax(col_has_data))
            col_hi = len(col_has_data) - int(np.argmax(col_has_data[::-1]))
            merged_db = merged_db[:, col_lo:col_hi]
        else:
            col_lo = 0

        # Nadir line: centre of the connector strip after outer crop
        nadir_col = port_cropped.shape[1] + gap_px // 2 - col_lo

        fig_w = min(MAX_WIDTH_PX / DPI, 14)
        fig_h = max(4, n_pings / 40)
        fig_h = min(fig_h, 80)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(np.ma.masked_invalid(merged_db),
                  aspect="auto", cmap="copper",
                  vmin=vmin, vmax=vmax, interpolation="none")
        ax.set_facecolor("#101010")

        ax.axvline(nadir_col - 0.5, color="cyan",
                   linewidth=0.8, alpha=0.7)

        ax.set_xlabel(
            f"← Port  |  Stbd →   (ground range, "
            f"0 to {WATERFALL_MAX_GROUND_M:.0f} m per side)   "
            f"[colour = backscatter dB, AGC-normalized]",
            fontsize=9,
        )
        ax.set_ylabel("Ping", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title(f"{freq} — {stem}", fontsize=10)
        plt.tight_layout()

        out_path = OUT_SSS / f"{stem}_{freq}.png"
        plt.savefig(out_path, dpi=DPI)
        plt.close(fig)

        valid_alt = altitudes[np.isfinite(altitudes)]
        results.append({
            "file": jsf_path.name,
            "freq": freq,
            "image": f"sss/{stem}_{freq}.png",
            "pings": int(n_pings),
            "ground_range_m": float(WATERFALL_MAX_GROUND_M),
            "ground_resolution_m": float(GROUND_RANGE_RESOLUTION_M),
            "altitude_min_m": float(valid_alt.min()) if valid_alt.size else None,
            "altitude_max_m": float(valid_alt.max()) if valid_alt.size else None,
            "altitude_median_m": float(np.median(valid_alt)) if valid_alt.size else None,
        })

    return results


def render_sbp(jsf_path):
    """Render SBP envelope profile."""
    try:
        data = read_sbp_jsf(jsf_path)
    except Exception as e:
        print(f"  Error reading {jsf_path.name}: {e}")
        return None

    if "SBP" not in data:
        return None

    amps = data["SBP"]["amps"]
    n_pings, n_samples = amps.shape
    if n_pings < 5:
        return None

    valid = amps[amps > 0]
    if len(valid) < 100:
        return None
    vmin, vmax = np.percentile(valid, [2, 95])

    fig_w = max(8, n_pings / 30)
    fig_w = min(fig_w, 60)
    fig_h = 5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(amps.T, aspect="auto", cmap="gray_r",
              vmin=vmin, vmax=vmax, interpolation="none",
              origin="upper")
    ax.set_xlabel("Ping (along-track)", fontsize=9)
    ax.set_ylabel("Sample (depth ↓)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(f"SBP — {jsf_path.stem}", fontsize=10)
    plt.tight_layout()

    out_path = OUT_SBP / f"{jsf_path.stem}.png"
    plt.savefig(out_path, dpi=DPI)
    plt.close(fig)

    return {
        "file": jsf_path.name,
        "image": f"sbp/{jsf_path.stem}.png",
        "pings": int(n_pings),
        "samples": int(n_samples),
    }


def main():
    ground_grid = np.arange(
        0.0, WATERFALL_MAX_GROUND_M + GROUND_RANGE_RESOLUTION_M,
        GROUND_RANGE_RESOLUTION_M,
        dtype=np.float32,
    )
    print(f"Ground-range grid: 0 to {WATERFALL_MAX_GROUND_M:.0f} m, "
          f"{GROUND_RANGE_RESOLUTION_M:.2f} m bins  "
          f"({len(ground_grid)} bins per side)")
    print(f"Kalman: Q={KALMAN_PROCESS_VAR}, R={KALMAN_MEASUREMENT_VAR}, "
          f"σ-gate={KALMAN_OUTLIER_SIGMA}")
    print(f"AGC: per-channel {GAIN_NORM_PERCENTILE}th-percentile reference")
    print(f"Inner crop: np.all (drops cols NaN in any ping)")
    print(f"Nadir connector: {NADIR_CONNECTOR_M:.1f} m visual strip + cyan line")

    index = {"sss": {}, "sbp": {}}

    sss_files = []
    for entry in sss_cfg["survey_dirs"]:
        d = ROOT / entry["path"]
        if d.exists():
            sss_files.extend(sorted(d.glob("*.jsf")))

    print(f"\nRendering SSS waterfalls ({len(sss_files)} files)...")
    for jsf in tqdm(sss_files, desc="SSS"):
        for r in render_sss(jsf, ground_grid):
            key = f"{jsf.name}_{r['freq']}"
            index["sss"][key] = r

    sbp_files = []
    for entry in sbp_cfg["survey_dirs"]:
        d = ROOT / entry["path"]
        if d.exists():
            sbp_files.extend(sorted(d.glob("*.jsf")))

    print(f"\nRendering SBP profiles ({len(sbp_files)} files)...")
    for jsf in tqdm(sbp_files, desc="SBP"):
        result = render_sbp(jsf)
        if result:
            index["sbp"][jsf.name] = result

    with open(OUT_INDEX, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone:")
    print(f"  SSS images: {len(index['sss'])}")
    print(f"  SBP images: {len(index['sbp'])}")
    print(f"  Index: {OUT_INDEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()