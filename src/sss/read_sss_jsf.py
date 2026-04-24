"""
EdgeTech JSF parser for SSS data (msg=80, subsys=20/21).

Heading source priority:
  1. msg=2020 attitude (always preferred — msg=80 heading field is
     unreliable on this instrument: bit3 is set but value is 0.0)
  2. msg=80 heading as fallback only if msg=2020 is absent AND
     the value is non-zero
"""
import struct
from pathlib import Path

import numpy as np

from src.config import get_config


def _nmea_to_decimal(value_str, hemisphere):
    try:
        val = float(value_str)
        degrees = int(val / 100)
        minutes = val - degrees * 100
        decimal = degrees + minutes / 60.0
        if hemisphere in ("S", "W"):
            decimal = -decimal
        return decimal
    except (ValueError, TypeError):
        return None


def _interp_clamp(t_query, t_known, v_known):
    if len(t_known) < 2:
        return np.full_like(t_query, np.nan)
    return np.interp(t_query, t_known, v_known, left=np.nan, right=np.nan)


def _interp_nan_outside(t_query, t_known, v_known):
    out = np.interp(t_query, t_known, v_known)
    out[t_query < t_known[0]] = np.nan
    out[t_query > t_known[-1]] = np.nan
    return out


def _interp_heading(t_query, t_known, v_known):
    sin_v = np.sin(np.radians(v_known))
    cos_v = np.cos(np.radians(v_known))
    sin_i = np.interp(t_query, t_known, sin_v)
    cos_i = np.interp(t_query, t_known, cos_v)
    return np.degrees(np.arctan2(sin_i, cos_i)) % 360


def read_sss_jsf(jsf_file_path, verbose=False):
    """Parse EdgeTech JSF. Returns dict with keys LF_port, LF_stbd, HF_port, HF_stbd."""
    jsf_path = Path(jsf_file_path)
    if not jsf_path.exists():
        raise FileNotFoundError(f"file not found: {jsf_path}")

    sound_speed = get_config()["environment"]["sound_speed"]

    raw = {
        ch: {
            "time": [], "amps": [], "pix_m": [], "center_freq_hz": [],
            "lat": [], "lon": [], "depth_m": [], "heave_m": [],
            "pitch": [], "roll": [], "heading": [],
        }
        for ch in ["LF_port", "LF_stbd", "HF_port", "HF_stbd"]
    }
    att_fallback = {"time": [], "pitch": [], "roll": [], "heading": [], "heave_m": []}
    nav_nmea = {"time": [], "lat": [], "lon": []}

    file_size = jsf_path.stat().st_size

    with open(jsf_path, "rb") as f:
        while f.tell() < file_size:
            hdr = f.read(16)
            if len(hdr) < 16:
                break
            if struct.unpack_from("<H", hdr, 0)[0] != 0x1601:
                f.seek(-15, 1)
                continue

            msg_type = struct.unpack_from("<H", hdr, 4)[0]
            subsys   = hdr[7]
            channel  = hdr[8]
            pay_size = struct.unpack_from("<I", hdr, 12)[0]
            pay      = f.read(pay_size)

            # ── SSS acoustic data ─────────────────────────
            if msg_type == 80 and subsys in (20, 21) and pay_size >= 240:
                t = (struct.unpack_from("<I", pay, 0)[0]
                     + struct.unpack_from("<I", pay, 200)[0] / 1_000_000.0)

                validity = struct.unpack_from("<H", pay, 30)[0]
                weight   = struct.unpack_from("<h", pay, 168)[0]

                ns    = struct.unpack_from("<I", pay, 116)[0]
                pix_m = sound_speed * ns * 1e-9 / 2.0

                center_freq = struct.unpack_from("<f", pay, 152)[0]
                if center_freq <= 0:
                    start_f = struct.unpack_from("<H", pay, 126)[0] * 10
                    end_f   = struct.unpack_from("<H", pay, 128)[0] * 10
                    center_freq = (start_f + end_f) / 2.0

                rx    = struct.unpack_from("<i", pay, 80)[0]
                ry    = struct.unpack_from("<i", pay, 84)[0]
                units = struct.unpack_from("<h", pay, 88)[0]
                lon, lat = np.nan, np.nan
                if validity & 0x0001 and rx != 0:
                    if units == 2:
                        lon = rx / 600000.0
                        lat = ry / 600000.0
                    elif units == 1:
                        lon = float(rx) / 1000.0
                        lat = float(ry) / 1000.0

                depth_m = 0.0
                if validity & (1 << 4):
                    depth_m = struct.unpack_from("<i", pay, 136)[0] / 1000.0

                heave_m = 0.0
                if validity & (1 << 7):
                    heave_m = struct.unpack_from("<f", pay, 48)[0]

                pitch, roll, heading = np.nan, np.nan, np.nan

                raw_u16 = np.frombuffer(pay[240:pay_size], dtype=np.uint16)
                amps = raw_u16.astype(np.float32) * (2.0 ** -weight)

                key = ("LF" if subsys == 20 else "HF") + ("_port" if channel == 0 else "_stbd")
                r = raw[key]
                r["time"].append(t)
                r["amps"].append(amps)
                r["pix_m"].append(pix_m)
                r["center_freq_hz"].append(center_freq)
                r["lat"].append(lat)
                r["lon"].append(lon)
                r["depth_m"].append(depth_m)
                r["heave_m"].append(heave_m)
                r["pitch"].append(pitch)
                r["roll"].append(roll)
                r["heading"].append(heading)

            # ── Attitude (msg=2020) ───────────────────────
            elif msg_type == 2020 and pay_size >= 40:
                t = (struct.unpack_from("<I", pay, 0)[0]
                     + struct.unpack_from("<I", pay, 4)[0] / 1000.0)
                validity = struct.unpack_from("<i", pay, 36)[0]
                if validity & (1 << 6) and validity & (1 << 7):
                    att_fallback["time"].append(t)
                    att_fallback["pitch"].append(
                        struct.unpack_from("<h", pay, 24)[0] * (180.0 / 32768.0))
                    att_fallback["roll"].append(
                        struct.unpack_from("<h", pay, 26)[0] * (180.0 / 32768.0))
                    att_fallback["heave_m"].append(
                        struct.unpack_from("<h", pay, 32)[0] / 1000.0)
                    att_fallback["heading"].append(
                        struct.unpack_from("<H", pay, 34)[0] / 100.0)

            # ── NMEA (msg=2002) ───────────────────────────
            elif msg_type == 2002 and pay_size > 12:
                t = (struct.unpack_from("<I", pay, 0)[0]
                     + struct.unpack_from("<I", pay, 4)[0] / 1000.0)
                nmea = pay[12:].decode("ascii", errors="ignore").strip()
                parts = nmea.split(",")
                lat, lon = None, None
                if "GGA" in parts[0] and len(parts) >= 6:
                    lat = _nmea_to_decimal(parts[2], parts[3])
                    lon = _nmea_to_decimal(parts[4], parts[5])
                elif "RMC" in parts[0] and len(parts) >= 9:
                    lat = _nmea_to_decimal(parts[3], parts[4])
                    lon = _nmea_to_decimal(parts[5], parts[6])
                if lat is not None:
                    nav_nmea["time"].append(t)
                    nav_nmea["lat"].append(lat)
                    nav_nmea["lon"].append(lon)

    # ── Assemble output ───────────────────────────────────
    std = {}
    for ch in ["LF_port", "LF_stbd", "HF_port", "HF_stbd"]:
        r = raw[ch]
        if not r["time"]:
            continue

        t_arr = np.array(r["time"])
        max_len = max(len(a) for a in r["amps"])
        amps_2d = np.array(
            [np.pad(a, (0, max_len - len(a))) for a in r["amps"]], dtype=np.float32)

        pitch_arr   = np.array(r["pitch"],   dtype=np.float64)
        roll_arr    = np.array(r["roll"],    dtype=np.float64)
        heading_arr = np.array(r["heading"], dtype=np.float64)

        if len(att_fallback["time"]) >= 2:
            at = np.array(att_fallback["time"])
            nan_p = np.isnan(pitch_arr)
            nan_r = np.isnan(roll_arr)
            nan_h = np.isnan(heading_arr)
            if nan_p.any():
                pitch_arr[nan_p] = _interp_clamp(
                    t_arr[nan_p], at, np.array(att_fallback["pitch"]))[nan_p]
            if nan_r.any():
                roll_arr[nan_r] = _interp_clamp(
                    t_arr[nan_r], at, np.array(att_fallback["roll"]))[nan_r]
            if nan_h.any():
                heading_arr[nan_h] = _interp_heading(
                    t_arr[nan_h], at, np.array(att_fallback["heading"]))[nan_h]

        lat_arr = np.array(r["lat"], dtype=np.float64)
        lon_arr = np.array(r["lon"], dtype=np.float64)

        if len(nav_nmea["time"]) >= 2:
            nt = np.array(nav_nmea["time"])
            order = np.argsort(nt)
            nt = nt[order]
            nl = np.array(nav_nmea["lat"])[order]
            no = np.array(nav_nmea["lon"])[order]
            lat_nav = _interp_nan_outside(t_arr, nt, nl)
            lon_nav = _interp_nan_outside(t_arr, nt, no)
            nan_mask = np.isnan(lat_arr)
            lat_arr[nan_mask] = lat_nav[nan_mask]
            lon_arr[nan_mask] = lon_nav[nan_mask]

        std[ch] = {
            "ping_time":      t_arr,
            "amps":           amps_2d,
            "pix_m":          np.array(r["pix_m"]),
            "center_freq_hz": np.array(r["center_freq_hz"]),
            "depth_m":        np.array(r["depth_m"]),
            "heave_m":        np.array(r["heave_m"]),
            "pitch":          pitch_arr,
            "roll":           roll_arr,
            "heading":        heading_arr,
            "lat":            lat_arr,
            "lon":            lon_arr,
            "pos_source":     np.where(~np.isnan(lat_arr), 0, 255).astype(np.uint8),
        }

    if verbose:
        for ch, cd in std.items():
            valid = ~np.isnan(cd["lat"])
            print(f"=== SSS {ch}: {jsf_path.name} ===")
            print(f"  Pings          : {len(cd['ping_time'])}")
            print(f"  pix_m          : {cd['pix_m'].mean():.4f} m")
            print(f"  center_freq_hz : {cd['center_freq_hz'].mean():.0f} Hz")
            print(f"  depth_m        : {cd['depth_m'].mean():.3f} m")
            valid_h = cd["heading"][~np.isnan(cd["heading"])]
            if valid_h.size:
                print(f"  heading range  : {valid_h.min():.1f} ~ {valid_h.max():.1f} deg")
            print(f"  GPS valid      : {valid.sum()} / {len(cd['ping_time'])}")

    return std