# src/data_loader/read_sbp_jsf.py
import struct
import numpy as np
from pathlib import Path


def _nmea_to_decimal(value_str, hemisphere):
    try:
        val = float(value_str)
        degrees = int(val / 100)
        minutes = val - degrees * 100
        decimal = degrees + minutes / 60.0
        if hemisphere in ('S', 'W'):
            decimal = -decimal
        return decimal
    except (ValueError, TypeError):
        return None


def _interp_clamp(t_query, t_known, v_known):
    if len(t_known) < 2:
        return np.full_like(t_query, np.nan)
    return np.interp(t_query, t_known, v_known,
                     left=np.nan, right=np.nan)


def _interp_nan_outside(t_query, t_known, v_known):
    out = np.interp(t_query, t_known, v_known)
    out[t_query < t_known[0]]  = np.nan
    out[t_query > t_known[-1]] = np.nan
    return out


def _interp_heading(t_query, t_known, v_known):
    sin_v = np.sin(np.radians(v_known))
    cos_v = np.cos(np.radians(v_known))
    sin_i = np.interp(t_query, t_known, sin_v)
    cos_i = np.interp(t_query, t_known, cos_v)
    return np.degrees(np.arctan2(sin_i, cos_i)) % 360


def read_sbp_jsf(jsf_file_path, verbose=False):
    """
    Parse EdgeTech JSF files for SBP data (msg=80).
    Data format 1: matched-filtered I/Q (two shorts per sample).
    All chirp parameters read directly from msg=80 header.

    Returns dict with key 'SBP' containing:
      ping_time, amps, samples_per_ping,
      start_freq_hz, end_freq_hz, sweep_ms, bw_hz,
      data_format, ns_per_sample,
      pitch, roll, heave_m,
      depth_m (transducer depth),
      altitude_m (transducer to seabed),
      water_depth_m (depth + altitude),
      lat, lon, pos_source
    """
    jsf_path = Path(jsf_file_path)
    if not jsf_path.exists():
        raise FileNotFoundError(f"file not found: {jsf_path}")

    raw = {
        "sbp": {
            "time": [], "amps": [], "n_samples": [],
            "start_freq_hz": [], "end_freq_hz": [],
            "sweep_ms": [], "data_format": [],
            "ns_per_sample": [],
            "heave_m": [], "depth_m": [], "altitude_m": [],
            "lat": [], "lon": [],
        },
        "attitude": {
            "time": [], "pitch": [], "roll": [],
            "heading": [], "heave_m": [],
        },
        "nav": {"time": [], "lat": [], "lon": []},
    }

    file_size = jsf_path.stat().st_size
    counter   = 0

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
            pay_size = struct.unpack_from("<I", hdr, 12)[0]
            pay      = f.read(pay_size)

            # ── SBP acoustic data ────────────────────────────────────
            if msg_type == 80 and subsys == 0 and pay_size >= 240:

                # Time
                t = (struct.unpack_from("<I", pay, 0)[0]
                     + struct.unpack_from("<I", pay, 200)[0] / 1_000_000.0)

                # MSB extension fields
                msb  = struct.unpack_from("<H", pay, 16)[0]
                lsb2 = struct.unpack_from("<H", pay, 20)[0]

                # Chirp parameters
                start_dahz  = struct.unpack_from("<H", pay, 126)[0]
                end_dahz    = struct.unpack_from("<H", pay, 128)[0]
                sweep_ms_i  = struct.unpack_from("<H", pay, 130)[0]
                start_msb   = (msb & 0x000F)
                end_msb     = (msb & 0x00F0) >> 4
                start_freq  = ((start_msb << 16) | start_dahz) * 10  # Hz
                end_freq    = ((end_msb   << 16) | end_dahz)   * 10  # Hz
                sweep_us    = (lsb2 >> 4) & 0x3FF
                sweep_ms    = sweep_ms_i + sweep_us / 1000.0

                # Sample interval
                ns_per_sample = struct.unpack_from("<I", pay, 116)[0]

                # Data format
                data_fmt = struct.unpack_from("<h", pay, 34)[0]

                # Validity flag
                validity = struct.unpack_from("<H", pay, 30)[0]

                # Navigation (lon/lat from msg=80)
                rx    = struct.unpack_from("<i", pay, 80)[0]
                ry    = struct.unpack_from("<i", pay, 84)[0]
                units = struct.unpack_from("<h", pay, 88)[0]
                lon, lat = 0.0, 0.0
                if validity & 0x0001:  # lat/lon valid
                    if units == 2 and rx != 0:
                        lon = rx / 600000.0
                        lat = ry / 600000.0
                    elif units == 1 and rx != 0:
                        lon = float(rx) / 1000.0
                        lat = float(ry) / 1000.0

                # Heave (from msg=80 directly)
                heave_m = 0.0
                if validity & (1 << 7):  # heave valid
                    heave_m = struct.unpack_from("<f", pay, 48)[0]

                # Transducer depth (from pressure)
                depth_m = 0.0
                if validity & (1 << 9):  # depth valid
                    depth_mm = struct.unpack_from("<i", pay, 136)[0]
                    depth_m  = depth_mm / 1000.0

                # Altitude (transducer to seabed)
                altitude_m = np.nan
                if validity & (1 << 6):  # altitude valid
                    alt_mm = struct.unpack_from("<i", pay, 144)[0]
                    if alt_mm > 0:
                        altitude_m = alt_mm / 1000.0

                # Acoustic samples
                data_len = pay_size - 240
                envelope = np.empty(0, dtype=np.float32)
                if data_len > 0:
                    raw_bytes = np.frombuffer(
                        pay[240:pay_size], dtype=np.int16)
                    if data_fmt in (1, 9):
                        # I/Q: two shorts per sample
                        try:
                            cplx = raw_bytes.reshape(-1, 2)
                            envelope = np.hypot(
                                cplx[:, 0].astype(np.float32),
                                cplx[:, 1].astype(np.float32))
                        except ValueError:
                            envelope = np.abs(
                                raw_bytes).astype(np.float32)
                    else:
                        # Envelope or raw: one short per sample
                        envelope = np.abs(
                            raw_bytes).astype(np.float32)

                ping_t = t if t > 0 else float(counter)
                s = raw["sbp"]
                s["time"].append(ping_t)
                s["amps"].append(envelope)
                s["n_samples"].append(len(envelope))
                s["start_freq_hz"].append(start_freq)
                s["end_freq_hz"].append(end_freq)
                s["sweep_ms"].append(sweep_ms)
                s["data_format"].append(data_fmt)
                s["ns_per_sample"].append(ns_per_sample)
                s["heave_m"].append(heave_m)
                s["depth_m"].append(depth_m)
                s["altitude_m"].append(altitude_m)
                s["lat"].append(lat if lat != 0.0 else np.nan)
                s["lon"].append(lon if lon != 0.0 else np.nan)
                counter += 1

            # ── Attitude (msg=2020) ──────────────────────────────────
            elif msg_type == 2020 and pay_size >= 40:
                t = (struct.unpack_from("<I", pay, 0)[0]
                     + struct.unpack_from("<I", pay, 4)[0] / 1000.0)
                validity = struct.unpack_from("<i", pay, 36)[0]
                if validity & (1 << 6) and validity & (1 << 7):
                    raw["attitude"]["time"].append(t)
                    raw["attitude"]["pitch"].append(
                        struct.unpack_from("<h", pay, 24)[0]
                        * (180.0 / 32768.0))
                    raw["attitude"]["roll"].append(
                        struct.unpack_from("<h", pay, 26)[0]
                        * (180.0 / 32768.0))
                    raw["attitude"]["heave_m"].append(
                        struct.unpack_from("<h", pay, 32)[0] / 1000.0)
                    raw["attitude"]["heading"].append(
                        struct.unpack_from("<H", pay, 34)[0] / 100.0)

            # ── NMEA navigation (msg=2002) ───────────────────────────
            elif msg_type == 2002 and pay_size > 12:
                t = (struct.unpack_from("<I", pay, 0)[0]
                     + struct.unpack_from("<I", pay, 4)[0] / 1000.0)
                nmea   = pay[12:].decode("ascii", errors="ignore").strip()
                parts  = nmea.split(",")
                lat, lon = None, None
                if "GGA" in parts[0] and len(parts) >= 6:
                    lat = _nmea_to_decimal(parts[2], parts[3])
                    lon = _nmea_to_decimal(parts[4], parts[5])
                elif "RMC" in parts[0] and len(parts) >= 9:
                    lat = _nmea_to_decimal(parts[3], parts[4])
                    lon = _nmea_to_decimal(parts[5], parts[6])
                if lat is not None:
                    raw["nav"]["time"].append(t)
                    raw["nav"]["lat"].append(lat)
                    raw["nav"]["lon"].append(lon)

    t_arr = np.array(raw["sbp"]["time"])
    if len(t_arr) == 0:
        return {}

    max_len  = max(len(a) for a in raw["sbp"]["amps"])
    amps_2d  = np.array(
        [np.pad(a, (0, max_len - len(a)))
         for a in raw["sbp"]["amps"]], dtype=np.float32)

    # Per-ping scalar fields
    s = raw["sbp"]
    altitude_arr = np.array(s["altitude_m"], dtype=np.float64)
    depth_arr    = np.array(s["depth_m"],    dtype=np.float64)
    heave_arr    = np.array(s["heave_m"],    dtype=np.float64)

    # water_depth = transducer depth + altitude
    water_depth  = np.where(
        np.isfinite(altitude_arr),
        depth_arr + altitude_arr,
        np.nan)

    cd = {
        "ping_time":      t_arr,
        "amps":           amps_2d,
        "samples_per_ping": np.array(s["n_samples"]),
        "start_freq_hz":  np.array(s["start_freq_hz"]),
        "end_freq_hz":    np.array(s["end_freq_hz"]),
        "sweep_ms":       np.array(s["sweep_ms"]),
        "data_format":    np.array(s["data_format"]),
        "ns_per_sample":  np.array(s["ns_per_sample"]),
        "heave_m":        heave_arr,
        "depth_m":        depth_arr,        # transducer depth
        "altitude_m":     altitude_arr,     # transducer to seabed
        "water_depth_m":  water_depth,      # total water depth
        "pitch":          np.full_like(t_arr, np.nan),
        "roll":           np.full_like(t_arr, np.nan),
        "heading":        np.full_like(t_arr, np.nan),
        "lat":            np.array(s["lat"]),
        "lon":            np.array(s["lon"]),
        "pos_source":     np.zeros(len(t_arr), dtype=np.uint8),
    }

    # Interpolate attitude from msg=2020
    att = raw["attitude"]
    if len(att["time"]) >= 2:
        at = np.array(att["time"])
        cd["pitch"]   = _interp_clamp(t_arr, at, np.array(att["pitch"]))
        cd["roll"]    = _interp_clamp(t_arr, at, np.array(att["roll"]))
        cd["heading"] = _interp_heading(
            t_arr, at, np.array(att["heading"]))

    # Fill lat/lon gaps with NMEA interpolation
    nav = raw["nav"]
    if len(nav["time"]) >= 2:
        nt    = np.array(nav["time"])
        order = np.argsort(nt)
        nt    = nt[order]
        nl    = np.array(nav["lat"])[order]
        no    = np.array(nav["lon"])[order]
        lat_nav = _interp_nan_outside(t_arr, nt, nl)
        lon_nav = _interp_nan_outside(t_arr, nt, no)
        # Use msg=80 nav where available, fill gaps with NMEA
        nan_mask = np.isnan(cd["lat"])
        cd["lat"][nan_mask] = lat_nav[nan_mask]
        cd["lon"][nan_mask] = lon_nav[nan_mask]

    if verbose:
        valid = ~np.isnan(cd["lon"])
        if valid.sum() > 1:
            lats = cd["lat"][valid]
            lons = cd["lon"][valid]
            dlat = np.diff(lats) * 111320
            dlon = (np.diff(lons)
                    * 111320 * np.cos(np.radians(lats[:-1])))
            spacing = np.sqrt(dlat**2 + dlon**2)
            wd = cd["water_depth_m"]
            print(f"=== SBP: {jsf_path.name} ===")
            print(f"Pings          : {len(t_arr)}")
            print(f"Data format    : {cd['data_format'][0]}")
            print(f"Start freq     : {cd['start_freq_hz'][0]} Hz")
            print(f"End freq       : {cd['end_freq_hz'][0]} Hz")
            print(f"BW             : "
                  f"{cd['end_freq_hz'][0]-cd['start_freq_hz'][0]} Hz")
            print(f"Sweep          : {cd['sweep_ms'][0]:.1f} ms")
            print(f"ns/sample      : {cd['ns_per_sample'][0]} ns")
            print(f"Ping spacing   : mean={spacing.mean():.2f}m")
            print(f"Water depth    : "
                  f"{np.nanmin(wd):.2f}~{np.nanmax(wd):.2f}m")
            print(f"Heave          : "
                  f"{cd['heave_m'].mean():.4f}m")
            print(f"Attitude recs  : {len(att['time'])}")

    return {"SBP": cd}