"""
src/config.py
Site-specific configuration.
Update SOUND_SPEED for each survey site based on SVP cast.
"""
from pathlib import Path

ROOT = Path(__file__).parent.parent  # src/ → project root

# Measured sound velocity (depth-weighted average from SVP)
# Mudan Reservoir, 2025-12-22/23
SOUND_SPEED = 1489.073

# Coordinate reference system
EPSG = 3826  # TWD97 TM2

# Grid resolution
RESOLUTION = 0.5  # m

# Water properties for TVG correction
WATER_TEMP_C  = 25.0   # °C, default if not measured
WATER_SALINITY = 0.0   # ppt, 0 = freshwater, ~35 = seawater

SBP_CC = 1.642613e+08
