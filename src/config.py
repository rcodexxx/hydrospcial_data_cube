"""
src/config.py

Global and site-specific configuration parameters for the Hydrospatial Data Cube pipeline.
Note: Environmental parameters (e.g., SOUND_SPEED) should be updated for each survey site
based on the local Sound Velocity Profile (SVP) cast.
"""
from pathlib import Path

# ==========================================
# 1. Project Structure
# ==========================================
# Defines the absolute path to the project root directory (two levels up from src/)
ROOT = Path(__file__).parent.parent


# ==========================================
# 2. Geospatial & Gridding Parameters
# ==========================================
# Coordinate Reference System (CRS) EPSG code
# Default: 3826 (TWD97 / TM2 zone 121, common for Taiwan inland waters)
EPSG = 3826

# Spatial resolution for the output GeoTIFF grids (in meters)
RESOLUTION = 0.5


# ==========================================
# 3. Environmental Parameters (Site-Specific)
# ==========================================
# Measured sound velocity (depth-weighted average from SVP)
# Current Survey: Mudan Reservoir, 2025-12-22/23
SOUND_SPEED = 1489.073

# Water properties for acoustic absorption and Time-Variable Gain (TVG) calculations
WATER_TEMP_C = 25.0    # Water temperature in Celsius (default if unmeasured)
WATER_SALINITY = 0.0   # Salinity in parts per thousand (ppt). 0 = freshwater, ~35 = seawater


# ==========================================
# 4. Acoustic Processing Parameters
# ==========================================
# Sub-Bottom Profiler (SBP) Calibration Constant
# Used for absolute/relative reflection coefficient or specific radiometric conversions.
SBP_CC = 1.642613e+08

# Side Scan Sonar (SSS) Reference Digital Number (Pseudo-Calibration)
# Represents the theoretical maximum of the 16-bit unsigned integer ADC (2^16).
# Used as the denominator when converting raw linear amplitude to relative Decibels (dB).
# Formula: 20 * log10(amplitude / REFERENCE_DN)
# This ensures output BS values fall within a physically realistic negative range (e.g., -80 to 0 dB).
REFERENCE_DN = 65536.0

REFERENCE_ANGLE = 45