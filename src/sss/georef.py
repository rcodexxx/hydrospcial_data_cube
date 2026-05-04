"""
Per-channel SSS georeferencing.

For one .jsf channel:
  - Filter out pings during turns / high roll / rapid heading change
  - Determine towfish altitude (first-return preferred, MBES fallback)
  - Apply median-filter altitude outlier rejection
  - Project each along-track sample to ground coordinates
  - Compute incidence angle per sample

Returns a dict of 1D arrays (one entry per valid sample).
"""
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol
from scipy.signal import medfilt

from src.config import get_config
from src.sss.config import (
    TURN_HEADING_THRESHOLD_DEG,
    ROLL_THRESHOLD_DEG,
    HEADING_RATE_THRESHOLD_DEG_S,
    ALTITUDE_MEDIAN_FILTER_WINDOW,
    ALTITUDE_OUTLIER_RATIO,
    get_far_cutoff,
    NADIR_CUTOFF_DEG
)
from src.sss.read_sss_jsf import read_sss_jsf

_EARTH_RADIUS_M = 6_371_000.0


# ──────────────────────────────────────────────────────────────
# First-return detection
# ──────────────────────────────────────────────────────────────
def detect_first_return(amps, pix_m, min_range_m=3.0, max_range_m=None,
                       threshold_ratio=0.1):
    """Threshold-based first-return detection (original version)."""
    min_idx = int(min_range_m / pix_m)
    if max_range_m is not None:
        max_idx = min(int(max_range_m / pix_m), len(amps))
    else:
        max_idx = len(amps)
    if min_idx >= max_idx:
        return None
    
    search = medfilt(amps[min_idx:max_idx].astype(np.float64), kernel_size=21)
    max_val = search.max()
    if max_val <= 0:
        return None
    threshold = max_val * threshold_ratio
    diff = np.diff(search)
    candidates = np.where(diff > threshold)[0]
    if len(candidates) == 0:
        return None
    return float((candidates[0] + min_idx) * pix_m)


# ──────────────────────────────────────────────────────────────
# Coordinate helpers
# ──────────────────────────────────────────────────────────────
def _offset_latlon(lat, lon, bearing_deg, dist_m):
    b = np.deg2rad(bearing_deg)
    return (
        lat + np.rad2deg(dist_m * np.cos(b) / _EARTH_RADIUS_M),
        lon + np.rad2deg(dist_m * np.sin(b) /
                         (_EARTH_RADIUS_M * np.cos(np.deg2rad(lat)))),
    )


def _query_mbes_batch(lats, lons, mbes_data, tf, tr):
    xs, ys = tr.transform(lons, lats)
    rows, cols = rowcol(tf, xs, ys)
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)

    valid = (rows >= 0) & (rows < mbes_data.shape[0]) & \
            (cols >= 0) & (cols < mbes_data.shape[1])

    result = np.full(len(lats), np.nan, dtype=np.float32)
    result[valid] = mbes_data[rows[valid], cols[valid]]
    return result


# ──────────────────────────────────────────────────────────────
# Ping-level turn filter
# ──────────────────────────────────────────────────────────────
def _turn_filter_mask(cd):
    """Return boolean mask of pings to keep (drop turns, high roll)."""
    n = len(cd["ping_time"])
    keep = np.zeros(n, dtype=bool)
    prev_heading = None
    prev_time = None

    for i in range(n):
        if cd["pos_source"][i] == 255:
            continue

        heading = float(cd["heading"][i])
        if np.isnan(heading):
            continue

        t = float(cd["ping_time"][i])
        roll = float(cd["roll"][i])

        if np.isfinite(roll) and abs(roll) > ROLL_THRESHOLD_DEG:
            prev_heading = heading
            prev_time = t
            continue

        if prev_heading is not None and prev_time is not None:
            dt = t - prev_time
            if dt > 0:
                dh = abs(heading - prev_heading)
                dh = min(dh, 360 - dh)
                if dh / dt > HEADING_RATE_THRESHOLD_DEG_S:
                    prev_heading = heading
                    prev_time = t
                    continue

        if prev_heading is not None:
            dh = min(abs(heading - prev_heading),
                     360 - abs(heading - prev_heading))
            if dh > TURN_HEADING_THRESHOLD_DEG:
                prev_heading = heading
                prev_time = t
                continue

        prev_heading = heading
        prev_time = t
        keep[i] = True

    return keep


# ──────────────────────────────────────────────────────────────
# Altitude determination
# ──────────────────────────────────────────────────────────────
def _compute_altitudes(idx, cd, mbes_zs, max_altitude, min_altitude=5.0):
    """
    Fbr-primary altitude. MBES used only as sanity check and fallback.
    
    Rationale: mbes_depth - towfish_depth has accumulated error from
    towfish position estimation, MBES grid interpolation, and depth
    sensor calibration. Direct acoustic first-return is more reliable.
    """
    altitudes = np.full(len(idx), np.nan, dtype=np.float32)
    
    for j, i in enumerate(idx):
        amps = cd["amps"][i].astype(np.float32)
        pix_m = float(cd["pix_m"][i])
        fbr = detect_first_return(amps, pix_m, max_range_m=max_altitude)
        
        depth_m = float(cd["depth_m"][i])
        mbes_z = float(mbes_zs[j]) if np.isfinite(mbes_zs[j]) else np.nan
        mbes_alt = (mbes_z - depth_m) if (np.isfinite(mbes_z) and depth_m > 0) else np.nan
        
        # fbr-primary: accept fbr if in valid range
        if fbr is not None and min_altitude < fbr < max_altitude:
            altitudes[j] = fbr
            continue
        
        # fbr failed: last resort is MBES-derived
        if np.isfinite(mbes_alt) and min_altitude < mbes_alt < max_altitude:
            altitudes[j] = mbes_alt
        # else: NaN, ping dropped
    
    return altitudes


def _filter_altitude_outliers(altitudes):
    """
    Reject along-track bottom-tracking glitches using median filter.
    Replace outliers (|alt - local_median| / local_median > threshold) with NaN.
    """
    altitudes = altitudes.copy()
    valid = np.isfinite(altitudes)
    if valid.sum() < ALTITUDE_MEDIAN_FILTER_WINDOW:
        return altitudes

    local_med = medfilt(
        np.where(valid, altitudes, np.median(altitudes[valid])),
        kernel_size=ALTITUDE_MEDIAN_FILTER_WINDOW,
    )
    ratio = np.abs(altitudes - local_med) / np.maximum(local_med, 1e-6)
    altitudes[ratio > ALTITUDE_OUTLIER_RATIO] = np.nan
    return altitudes


# ──────────────────────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────────────────────
def georef_line(jsf_path, mbes_tif, channel, cable_length=None, mbes_preloaded=None):
    """
    Georeference one channel of one .jsf file.

    Returns dict of 1D arrays, or None if no valid data:
      lat, lon, bs_linear, altitude, ground_m, slant_m, inc_angle,
      ping_idx, heading
    """
    data = read_sss_jsf(Path(jsf_path))
    if channel not in data:
        return None
    cd = data[channel]
    if np.isnan(cd["lat"]).all():
        return None

    # Load MBES
    if mbes_preloaded is not None:
        mbes_data = mbes_preloaded["data"]
        tf = mbes_preloaded["transform"]
        tr = mbes_preloaded["tr"]
    else:
        with rasterio.open(mbes_tif) as src:
            mbes_data = src.read(1).astype(np.float32)
            tf = src.transform
            tr = Transformer.from_crs(
                "EPSG:4326", f"EPSG:{src.crs.to_epsg()}", always_xy=True)

    side = -90.0 if "port" in channel else 90.0

    # ── Pass 1: filter pings ──────────────────────────────
    keep = _turn_filter_mask(cd)
    if not keep.any():
        return None

    idx = np.where(keep)[0]
    lats = cd["lat"][idx].astype(np.float64)
    lons = cd["lon"][idx].astype(np.float64)
    depths = cd["depth_m"][idx].astype(np.float64)
    headings = cd["heading"][idx].astype(np.float64)
    pix_ms = cd["pix_m"][idx].astype(np.float64)

    # Layback correction
    if cable_length is not None:
        laybacks = np.sqrt(np.maximum(cable_length**2 - depths**2, 0.0))
        lats, lons = _offset_latlon(lats, lons, headings + 180.0, laybacks)

    # MBES depth query
    valid_pos = ~np.isnan(lats) & (pix_ms > 0)
    mbes_zs = np.full(len(idx), np.nan, dtype=np.float32)
    if valid_pos.any():
        mbes_zs[valid_pos] = _query_mbes_batch(
            lats[valid_pos], lons[valid_pos], mbes_data, tf, tr)

    # ── Pass 2: altitude + outlier rejection ──────────────
    max_altitude = float(np.nanmax(mbes_data)) * 1.15
    altitudes = _compute_altitudes(idx, cd, mbes_zs, max_altitude)
    altitudes = _filter_altitude_outliers(altitudes)

    # ── Pass 3: per-ping flat-bottom projection ───────────
    out = {k: [] for k in (
        "lat", "lon", "bs_linear", "altitude", "ground_m",
        "slant_m", "inc_angle", "ping_idx", "heading")}
    ping_counter = 0

    for j, i in enumerate(idx):
        if not valid_pos[j]:
            continue
        altitude = altitudes[j]
        if not np.isfinite(altitude):
            continue

        pix_m = float(pix_ms[j])
        amps = cd["amps"][i].astype(np.float32)

        slant = np.arange(len(amps), dtype=np.float32) * pix_m
        far_cutoff = get_far_cutoff(channel)
        min_slant = altitude / np.cos(np.deg2rad(NADIR_CUTOFF_DEG))
        max_slant = altitude / np.cos(np.deg2rad(far_cutoff))
        mask = (slant > min_slant) & (slant < max_slant)
        if not mask.any():
            continue

        sv = slant[mask]
        gv = np.sqrt(np.maximum(sv**2 - altitude**2, 0.0)).astype(np.float32)
        inc_v = np.rad2deg(np.arccos(
            np.clip(altitude / sv, -1.0, 1.0))).astype(np.float32)

        lat, lon = float(lats[j]), float(lons[j])
        heading = float(headings[j])
        s_lats, s_lons = _offset_latlon(lat, lon, heading + side, gv)

        n = int(mask.sum())
        out["lat"].append(s_lats)
        out["lon"].append(s_lons)
        out["bs_linear"].append(amps[mask])
        out["altitude"].append(np.full(n, altitude, dtype=np.float32))
        out["ground_m"].append(gv)
        out["slant_m"].append(sv.astype(np.float32))
        out["inc_angle"].append(inc_v)
        out["ping_idx"].append(np.full(n, ping_counter, dtype=np.int32))
        out["heading"].append(np.full(n, heading, dtype=np.float32))
        ping_counter += 1

    if not out["bs_linear"]:
        return None

    return {k: np.concatenate(v) for k, v in out.items()}