"""
SeaSPY2 .mag file parser.

Returns REAL records only, coordinates converted from lon/lat to
the project EPSG (from yaml). The file's built-in XY columns are
ignored (wrong CRS).
"""
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
from pyproj import Transformer

from src.config import get_config

_PATTERN = re.compile(
    r"\*[\d.]+/([\d:]+\.\d+)\s+"     # 1: time
    r"F:([\d.]+)\s+"                  # 2: F_nT
    r"S:\d+\s+"                       # speed (skip)
    r"D:([+-][\d.]+)m\s+"             # 3: depth
    r"A:([\d.]+)m\s+"                 # 4: altitude
    r".*?Q:(\d+)\s+"                  # 5: quality
    r"X:[\d.]+\s+Y:[\d.]+.*?"         # file XY (wrong CRS, skip)
    r"x:([\d.]+)\s+y:([\d.]+)\s+"     # 6: lon, 7: lat
    r"<(\w+)>"                        # 8: REAL or INTERP
)

_transformer = None


def _get_transformer():
    """Lazy init transformer using project EPSG from yaml."""
    global _transformer
    if _transformer is None:
        epsg = get_config()["grid"]["epsg"]
        _transformer = Transformer.from_crs(
            "EPSG:4326", f"EPSG:{epsg}", always_xy=True
        )
    return _transformer


def read_mag(file_path, apply_layback: bool = True) -> List[Dict]:
    """
    Parse a SeaSPY2 .mag file.

    Parameters
    ----------
    file_path     : path to .mag file
    apply_layback : correct towfish position behind GPS along track
                    (layback distance from yaml)

    Returns
    -------
    records : list of dicts with keys
        time, F_nT, depth_m, alt_m, quality, x, y, lon, lat
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"file not found: {file_path}")

    transformer = _get_transformer()
    records = []

    with open(file_path, "r", errors="ignore") as f:
        for line in f:
            m = _PATTERN.search(line)
            if not m or m.group(8) != "REAL":
                continue

            lon = float(m.group(6))
            lat = float(m.group(7))
            x, y = transformer.transform(lon, lat)

            records.append({
                "time": m.group(1),
                "F_nT": float(m.group(2)),
                "depth_m": float(m.group(3)),
                "alt_m": float(m.group(4)),
                "quality": int(m.group(5)),
                "x": x, "y": y, "lon": lon, "lat": lat,
            })

    if not records or not apply_layback:
        return records

    # Layback: shift position behind GPS along instantaneous heading
    layback_m = float(get_config()["mag"]["layback_m"])
    xs = np.array([r["x"] for r in records])
    ys = np.array([r["y"] for r in records])
    dx = np.diff(xs)
    dy = np.diff(ys)
    hdg = np.arctan2(dx, dy)
    hdg = np.concatenate([[hdg[0]], hdg])

    for i, r in enumerate(records):
        r["x"] = float(xs[i] - layback_m * np.sin(hdg[i]))
        r["y"] = float(ys[i] - layback_m * np.cos(hdg[i]))

    return records