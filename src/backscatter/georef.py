# src/backscatter/georef.py
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol

from src.backscatter.correction import detect_first_return
from src.config import SOUND_SPEED
from src.data_loader.read_sss_jsf import read_sss_jsf

_EARTH_RADIUS_M = 6_371_000.0


def _query_mbes(lat, lon, mbes_data, tf, tr):
    try:
        x, y = tr.transform(float(lon), float(lat))
        r, c = rowcol(tf, x, y)
        r, c = int(r), int(c)
        if not (0 <= r < mbes_data.shape[0] and 0 <= c < mbes_data.shape[1]):
            return None
        z = float(mbes_data[r, c])
        return None if np.isnan(z) else z
    except Exception:
        return None


def _offset_latlon(lat, lon, bearing_deg, dist_m):
    b = np.deg2rad(bearing_deg)
    return (
        lat + np.rad2deg(dist_m * np.cos(b) / _EARTH_RADIUS_M),
        lon
        + np.rad2deg(dist_m * np.sin(b) / (_EARTH_RADIUS_M * np.cos(np.deg2rad(lat)))),
    )


def _insonified_area(slant_m, inc_angle_rad, pulse_width_s):
    """
    Compute insonified area for flat seafloor assumption.

    A(theta) = (c * tau * R) / (2 * cos(theta))

    where c = sound speed, tau = pulse width, R = slant range,
    theta = incidence angle.

    Returns area in m^2. Used to normalize raw amplitude to
    per-unit-area backscatter strength.
    """
    cos_theta = np.cos(inc_angle_rad)
    # avoid division by zero at grazing angles
    cos_theta = np.maximum(cos_theta, 0.01)
    return (SOUND_SPEED * pulse_width_s * slant_m) / (2.0 * cos_theta)

def _query_mbes_batch(lats, lons, mbes_data, tf, tr):
    """Batch query MBES depth for all pings at once."""
    xs, ys = tr.transform(lons, lats)
    rows, cols = rowcol(tf, xs, ys)
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    
    valid = (rows >= 0) & (rows < mbes_data.shape[0]) & \
            (cols >= 0) & (cols < mbes_data.shape[1])
    
    result = np.full(len(lats), np.nan, dtype=np.float32)
    result[valid] = mbes_data[rows[valid], cols[valid]]
    return result


def georef_line(
    jsf_path,
    mbes_tif,
    channel,
    cable_length=None,
    turn_threshold=5.0,
    turn_cooldown=0.0,
    roll_threshold=5.0,
    heading_rate_threshold=3.0,
    mbes_preloaded=None,
):
    data = read_sss_jsf(Path(jsf_path))
    if channel not in data:
        return None
    cd = data[channel]
    if np.isnan(cd["lat"]).all():
        return None

    if mbes_preloaded is not None:
        mbes_data = mbes_preloaded["data"]
        tf = mbes_preloaded["transform"]
        tr = mbes_preloaded["tr"]
    else:
        with rasterio.open(mbes_tif) as src:
            mbes_data = src.read(1).astype(np.float32)
            tf = src.transform
            tr = Transformer.from_crs(
                "EPSG:4326", f"EPSG:{src.crs.to_epsg()}", always_xy=True
            )

    center_freq = float(np.mean(cd["center_freq_hz"]))
    pulse_width_s = 10e-6 if center_freq > 500000 else 50e-6
    side = -90.0 if "port" in channel else 90.0

    # ── 第一段：turn filter + 批量altitude計算 ──────────────
    n_pings = len(cd["ping_time"])
    keep = np.zeros(n_pings, dtype=bool)
    last_turn_time = -np.inf
    prev_heading = None
    prev_time = None

    for i in range(n_pings):
        if cd["pos_source"][i] == 255:
            continue

        heading = float(cd["heading"][i])
        if np.isnan(heading):
            continue

        t = float(cd["ping_time"][i])

        # roll過濾
        roll = float(cd["roll"][i])
        if np.isfinite(roll) and abs(roll) > roll_threshold:
            last_turn_time = t
            prev_heading = heading
            prev_time = t
            continue

        # heading rate過濾（取代原本的turn_threshold）
        if prev_heading is not None and prev_time is not None:
            dt = t - prev_time
            if dt > 0:
                dh = abs(heading - prev_heading)
                dh = min(dh, 360 - dh)
                rate = dh / dt
                if rate > heading_rate_threshold:
                    last_turn_time = t
                    prev_heading = heading
                    prev_time = t
                    continue

        # turn_threshold保留作為備用（per-ping的heading差異）
        if prev_heading is not None:
            dh = min(abs(heading - prev_heading), 360 - abs(heading - prev_heading))
            if dh > turn_threshold:
                last_turn_time = t
                prev_heading = heading
                prev_time = t
                continue

        prev_heading = heading
        prev_time = t

        if t - last_turn_time < turn_cooldown:
            continue

        keep[i] = True

    if not keep.any():
        return None

    idx = np.where(keep)[0]
    lats = cd["lat"][idx].astype(np.float64)
    lons = cd["lon"][idx].astype(np.float64)
    depths = cd["depth_m"][idx].astype(np.float64)
    headings = cd["heading"][idx].astype(np.float64)
    pix_ms = cd["pix_m"][idx].astype(np.float64)

    # layback correction (vectorized)
    if cable_length is not None:
        laybacks = np.sqrt(np.maximum(cable_length**2 - depths**2, 0.0))
        lats, lons = _offset_latlon(lats, lons, headings + 180.0, laybacks)

    # 批量MBES查詢
    valid_pos = ~np.isnan(lats) & (pix_ms > 0)
    mbes_zs = np.full(len(idx), np.nan)
    if valid_pos.any():
        mbes_zs[valid_pos] = _query_mbes_batch(
            lats[valid_pos], lons[valid_pos], mbes_data, tf, tr
        )

    # ── 第二段：逐ping做slant-range投影 ──────────────
    out = {k: [] for k in ("lat", "lon", "bs", "altitude", "ground_m",
                            "slant_m", "inc_angle", "ping_idx", "heading")}
    ping_counter = 0

    for j, i in enumerate(idx):
        if not valid_pos[j]:
            continue

        pix_m = float(pix_ms[j])
        depth_m = float(depths[j])
        mbes_z = float(mbes_zs[j]) if not np.isnan(mbes_zs[j]) else None
        amps = cd["amps"][i].astype(np.float32)

        altitude = (mbes_z - depth_m) if (mbes_z is not None and depth_m > 0) else None
        fbr = detect_first_return(amps, pix_m)
        if fbr is not None:
            altitude = fbr if altitude is None else max(altitude, fbr)
        if altitude is None or altitude <= 0:
            continue

        slant = np.arange(len(amps), dtype=np.float32) * pix_m
        mask = slant > altitude
        if not mask.any():
            continue

        sv = slant[mask]
        gv = np.sqrt(np.maximum(sv**2 - altitude**2, 0.0))
        inc_v = np.rad2deg(np.arccos(np.clip(altitude / sv, -1.0, 1.0)))

        lat, lon = float(lats[j]), float(lons[j])
        heading = float(headings[j])
        s_lats, s_lons = _offset_latlon(lat, lon, heading + side, gv)

        n = int(mask.sum())
        out["lat"].append(s_lats)
        out["lon"].append(s_lons)
        out["bs"].append(amps[mask])
        out["altitude"].append(np.full(n, altitude, dtype=np.float32))
        out["ground_m"].append(gv.astype(np.float32))
        out["slant_m"].append(sv.astype(np.float32))
        out["inc_angle"].append(inc_v.astype(np.float32))
        out["ping_idx"].append(np.full(n, ping_counter, dtype=np.int32))
        out["heading"].append(np.full(n, heading, dtype=np.float32))
        ping_counter += 1

    if not out["bs"]:
        return None

    result = {k: np.concatenate(v) for k, v in out.items()}
    result["center_freq_hz"] = float(cd["center_freq_hz"].mean())
    return result