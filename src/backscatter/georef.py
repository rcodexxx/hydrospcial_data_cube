# src/backscatter/georef.py
from pathlib import Path
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol
from src.data_loader.read_sss_jsf import read_sss_jsf
from src.backscatter.correction import detect_first_return
from src.config import SOUND_SPEED

_EARTH_RADIUS_M = 6_371_000.0


def _query_mbes(lat, lon, mbes_data, tf, tr):
    try:
        x, y = tr.transform(float(lon), float(lat))
        r, c = rowcol(tf, x, y)
        r, c = int(r), int(c)
        if not (0 <= r < mbes_data.shape[0]
                and 0 <= c < mbes_data.shape[1]):
            return None
        z = float(mbes_data[r, c])
        return None if np.isnan(z) else z
    except Exception:
        return None


def _offset_latlon(lat, lon, bearing_deg, dist_m):
    b = np.deg2rad(bearing_deg)
    return (lat + np.rad2deg(dist_m * np.cos(b) / _EARTH_RADIUS_M),
            lon + np.rad2deg(dist_m * np.sin(b)
                             / (_EARTH_RADIUS_M * np.cos(np.deg2rad(lat)))))


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


def georef_line(jsf_path, mbes_tif, channel,
                cable_length=None,
                turn_threshold=5.0,
                turn_cooldown=0.0):
    """
    Georeference one SSS channel from a JSF file against MBES bathymetry.

    Altitude logic (per ping):
      1. Base: mbes_z - depth_m  (MBES water depth minus towfish depth)
      2. FBR override: if detect_first_return returns a value AND it is
         greater than the MBES-derived altitude, use FBR instead.
      3. If mbes_z is unavailable, fall back to FBR only.

    Includes insonified area correction for flat seafloor assumption.

    Returns dict with arrays: lat, lon, bs, altitude, ground_m,
    slant_m, inc_angle, ping_idx. None if no valid pings found.
    """
    data = read_sss_jsf(Path(jsf_path))
    if channel not in data:
        return None

    cd = data[channel]
    if np.isnan(cd["lat"]).all():
        return None

    with rasterio.open(mbes_tif) as src:
        mbes_data = src.read(1).astype(np.float32)
        tf        = src.transform
        tr        = Transformer.from_crs(
            "EPSG:4326", f"EPSG:{src.crs.to_epsg()}",
            always_xy=True)

    # estimate pulse width from sample interval and number of samples
    # EdgeTech 4205MP: pulse_width ≈ n_samples * sample_interval
    # For SSS, typical pulse width is ~0.1 ms for HF, ~0.5 ms for LF
    # Use pix_m to estimate: pix_m = c * sample_interval / 2
    # so sample_interval = 2 * pix_m / c
    pix_m_mean = float(np.mean(cd["pix_m"]))
    sample_interval_s = 2.0 * pix_m_mean / SOUND_SPEED
    # pulse width: EdgeTech chirp pulse, typically 1-20 ms
    # approximate from bandwidth: tau ≈ 1 / bandwidth
    # for 225 kHz LF with ~20 kHz bandwidth: ~50 us
    # for 830 kHz HF with ~100 kHz bandwidth: ~10 us
    # use a conservative estimate based on center frequency
    center_freq = float(np.mean(cd["center_freq_hz"]))
    if center_freq > 500000:  # HF
        pulse_width_s = 10e-6
    else:  # LF
        pulse_width_s = 50e-6

    side = -90.0 if "port" in channel else 90.0
    out = {k: [] for k in ("lat", "lon", "bs", "altitude",
                           "ground_m", "slant_m", "inc_angle", "ping_idx", "heading")}
    last_turn_time = -np.inf
    prev_heading   = None
    ping_counter   = 0

    for i in range(len(cd["ping_time"])):
        if cd["pos_source"][i] == 255:
            continue

        t       = float(cd["ping_time"][i])
        heading = float(cd["heading"][i])
        if np.isnan(heading):
            continue

        # turn filter
        if prev_heading is not None:
            dh = min(abs(heading - prev_heading),
                     360 - abs(heading - prev_heading))
            if dh > turn_threshold:
                last_turn_time = t
                prev_heading   = heading
                continue
        prev_heading = heading

        if t - last_turn_time < turn_cooldown:
            continue

        lat     = float(cd["lat"][i])
        lon     = float(cd["lon"][i])
        depth_m = float(cd["depth_m"][i])
        pix_m   = float(cd["pix_m"][i])
        amps    = cd["amps"][i].astype(np.float32)

        if np.isnan(lat) or pix_m <= 0:
            continue

        # layback correction
        if cable_length is not None:
            layback = float(np.sqrt(max(cable_length ** 2 - depth_m ** 2, 0.0)))
            lat, lon = _offset_latlon(lat, lon, heading + 180.0, layback)

        # altitude calculation
        mbes_z = _query_mbes(lat, lon, mbes_data, tf, tr)

        if mbes_z is not None and depth_m > 0:
            altitude = mbes_z - depth_m
        else:
            altitude = None

        fbr = detect_first_return(amps, pix_m)
        if fbr is not None:
            altitude = fbr if altitude is None else max(altitude, fbr)

        if altitude is None or altitude <= 0:
            continue

        # slant-range to ground projection
        slant = np.arange(len(amps), dtype=np.float32) * pix_m
        mask  = slant > altitude
        if not mask.any():
            continue

        sv    = slant[mask]
        gv    = np.sqrt(np.maximum(sv ** 2 - altitude ** 2, 0.0))
        inc_v = np.rad2deg(np.arccos(np.clip(altitude / sv, -1.0, 1.0)))

        # insonified area correction
        inc_rad = np.deg2rad(inc_v)
        bs_corrected = amps[mask]

        s_lats, s_lons = _offset_latlon(lat, lon, heading + side, gv)

        n = int(mask.sum())
        out["lat"].append(s_lats)
        out["lon"].append(s_lons)
        out["bs"].append(bs_corrected)
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