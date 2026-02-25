# data_loader.py
import struct
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def read_xyz(file_path) -> pd.DataFrame:
    """Reads standard whitespace-separated XYZ files (TWD97)."""
    return pd.read_csv(file_path, sep=r"\s+", header=None, names=["x", "y", "z"])


def read_jsf(file_path) -> List[Dict[str, any]]:
    """
    Parses EdgeTech JSF files (Msg 80 only).
    Returns list of dicts: {'ping', 'lon', 'lat', 'amps'}.
    """
    file_path = Path(file_path)
    pings_data = []

    if not file_path.exists():
        raise FileNotFoundError(f"找不到 JSF 檔案: {file_path}")

    file_size = file_path.stat().st_size
    counter = 0

    try:
        with open(file_path, "rb") as f:
            while f.tell() < file_size:
                # 1. Read packet header (16 bytes)
                header = f.read(16)
                if len(header) < 16:
                    break

                msg_type = struct.unpack_from("<H", header, 4)[0]
                packet_size = struct.unpack_from("<I", header, 12)[0]

                # 2. Process Sonar Data (Msg 80)
                if msg_type == 80 and packet_size >= 240:
                    type80_header = f.read(240)

                    # Extract metadata
                    num = struct.unpack_from("<I", type80_header, 4)[0]
                    rx = struct.unpack_from("<i", type80_header, 80)[0]
                    ry = struct.unpack_from("<i", type80_header, 84)[0]
                    units = struct.unpack_from("<h", type80_header, 88)[0]

                    if num == 0:
                        num = counter

                    counter += 1

                    # Coordinate scaling (Arcseconds to Degrees)
                    lon, lat = 0.0, 0.0
                    if units == 2 and rx != 0:
                        lon = rx / 600000.0
                        lat = ry / 600000.0
                    elif units == 1 and rx != 0:
                        lon = float(rx)
                        lat = float(ry)

                    # 3. Process Payload (Analytic Signal)
                    data_len = packet_size - 240
                    if data_len > 0:
                        raw_bytes = f.read(data_len)
                        raw_data = np.frombuffer(raw_bytes, dtype=np.int16)

                        # Compute Envelope: sqrt(Real^2 + Imag^2)
                        try:
                            complex_data = raw_data.reshape(-1, 2)
                            envelope = np.hypot(complex_data[:, 0], complex_data[:, 1])
                        except ValueError:
                            envelope = np.abs(raw_data)  # Fallback

                        pings_data.append(
                            {"ping": num, "lon": lon, "lat": lat, "amps": envelope}
                        )
                else:
                    # Skip other message types
                    f.seek(packet_size, 1)

    except Exception as e:
        raise RuntimeError(f"Error reading JSF({file_path}): {e}") from e

    return pings_data
