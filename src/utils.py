import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator
from pyproj import Transformer

CRS_TWD97 = "epsg:3826"
CRS_WGS84 = "epsg:4326"


def twd97_to_wgs84(x, y):
    """Transforms TWD97 coordinates to WGS84 (Lon/Lat)."""
    transformer = Transformer.from_crs(CRS_TWD97, CRS_WGS84, always_xy=True)
    return transformer.transform(x, y)


def wgs84_to_twd97(lon, lat):
    """Transforms WGS84 coordinates to TWD97 (X/Y)."""
    transformer = Transformer.from_crs(CRS_WGS84, CRS_TWD97, always_xy=True)
    return transformer.transform(lon, lat)


def _decimal_to_dms(deg, is_lat=False):
    if is_lat:
        direction = "N" if deg >= 0 else "S"
    else:
        direction = "E" if deg >= 0 else "W"

    deg = abs(deg)
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60) * 3600

    return f"{d}°{m:02d}'{round(s):02d}\"{direction}"


def apply_dms_ticks(ax, step_seconds=10.0):
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: _decimal_to_dms(x, is_lat=False))
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, pos: _decimal_to_dms(y, is_lat=True))
    )

    step_degrees = step_seconds / 3600.0
    ax.xaxis.set_major_locator(MultipleLocator(step_degrees))
    ax.yaxis.set_major_locator(MultipleLocator(step_degrees))


def set_map_aspect(ax, lats=None):
    if lats is not None and len(lats) > 0:
        mean_lat = np.mean(lats)
    else:
        mean_lat = 24.0

    aspect_ratio = 1.0 / np.cos(np.radians(mean_lat))
    ax.set_aspect(aspect_ratio)
