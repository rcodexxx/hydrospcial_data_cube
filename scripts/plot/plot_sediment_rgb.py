import numpy as np
import rasterio
from pathlib import Path

ROOT = Path(".")
SED_TIF = ROOT / "outputs/tif/sbp_sediment_class.tif"
OUT_TIF = ROOT / "outputs/tif/sbp_sediment_rgb.tif"

SED_COLORS = [
    (194, 166, 77),   # 0: Coarse sand
    (212, 185, 106),  # 1: Fine sand
    (220, 200, 138),  # 2: Very fine sand
    (184, 168, 138),  # 3: Silty sand
    (160, 152, 128),  # 4: Sandy silt
    (138, 133, 120),  # 5: Silt
    (122, 122, 112),  # 6: Sandy-silt-clay
    (107, 110, 106),  # 7: Silty clay
    (92, 98, 96),     # 8: Clayey silt
    (74, 85, 80),     # 9: Framework-supported mud
    (58, 74, 85),     # 10: Fluid mud
]

with rasterio.open(SED_TIF) as src:
    data = src.read(1)
    profile = src.profile.copy()

# create RGB image
h, w = data.shape
rgb = np.zeros((3, h, w), dtype=np.uint8)

for cls_id, (r, g, b) in enumerate(SED_COLORS):
    mask = data == cls_id
    rgb[0][mask] = r
    rgb[1][mask] = g
    rgb[2][mask] = b

# nodata = transparent (use alpha band)
alpha = np.where((data >= 0) & (data <= 10), 255, 0).astype(np.uint8)

profile.update(dtype='uint8', count=4, nodata=None)
with rasterio.open(OUT_TIF, 'w', **profile) as dst:
    dst.write(rgb[0], 1)
    dst.write(rgb[1], 2)
    dst.write(rgb[2], 3)
    dst.write(alpha, 4)

print(f"Saved: {OUT_TIF}")

# build overviews
with rasterio.open(OUT_TIF, 'r+') as ds:
    ds.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.nearest)
    ds.update_tags(ns='rio_overview', resampling='nearest')
print("Overviews built.")