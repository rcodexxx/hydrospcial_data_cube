# src/data_loader/read_mag.py
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
from pyproj import Transformer

_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True)
LAYBACK_M = 14.10

_PATTERN = re.compile(
    r"\*[\d.]+/([\d:]+\.\d+)\s+"  # 1: time
    r"F:([\d.]+)\s+"  # 2: F_nT
    r"S:\d+\s+"  # speed (skip)
    r"D:([+-][\d.]+)m\s+"  # 3: depth
    r"A:([\d.]+)m\s+"  # 4: altitude
    r".*?Q:(\d+)\s+"  # 5: quality
    r"X:[\d.]+\s+Y:[\d.]+.*?"  # file XY (skip, wrong CRS)
    r"x:([\d.]+)\s+y:([\d.]+)\s+"  # 6: lon, 7: lat
    r"<(\w+)>"  # 8: REAL or INTERP
)


def read_mag(file_path, apply_layback: bool = True) -> List[Dict]:
    """
    Parse SeaSPY2 .mag files.
    Returns REAL records only, coordinates converted to EPSG:3826 from lon/lat.
    The file's built-in XY columns are ignored (wrong CRS).

    Parameters
    ----------
    file_path      : path to .mag file
    apply_layback  : correct towfish position 14.1m behind GPS along track
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"file not found: {file_path}")

    records = []
    with open(file_path, "r", errors="ignore") as f:
        for line in f:
            m = _PATTERN.search(line)
            if not m:
                continue
            if m.group(8) != "REAL":
                continue

            lon = float(m.group(6))
            lat = float(m.group(7))
            x, y = _transformer.transform(lon, lat)

            records.append(
                {
                    "time": m.group(1),
                    "F_nT": float(m.group(2)),
                    "depth_m": float(m.group(3)),
                    "alt_m": float(m.group(4)),
                    "quality": int(m.group(5)),
                    "x": x,
                    "y": y,
                    "lon": lon,
                    "lat": lat,
                }
            )

    if not records or not apply_layback:
        return records

    # Layback correction: move position 14.1m behind GPS along track
    xs = np.array([r["x"] for r in records])
    ys = np.array([r["y"] for r in records])
    dx = np.diff(xs)
    dy = np.diff(ys)
    hdg = np.arctan2(dx, dy)
    hdg = np.concatenate([[hdg[0]], hdg])

    for i, r in enumerate(records):
        r["x"] = float(xs[i] - LAYBACK_M * np.sin(hdg[i]))
        r["y"] = float(ys[i] - LAYBACK_M * np.cos(hdg[i]))

    return records
