"""
Generate RGB GeoTIFF from categorical sediment classification.

Outputs a 3-band RGB GeoTIFF colored by SEDIMENT_COLORS palette,
with a designated nodata color (255, 255, 255) for non-coverage
pixels. Tile server treats this color as transparent.

Output is masked to MBES bathymetry coverage so the viewer never
shows SBP-only areas outside the multibeam survey extent.
"""
import numpy as np
import rasterio

from src.config import ROOT, get_config
from src.sbp.config import SEDIMENT_COLORS

# Sentinel RGB for transparent areas. White (255,255,255) chosen
# because it never appears in the natural sediment palette.
NODATA_RGB = (255, 255, 255)


def hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def main():
    cfg = get_config()
    src_path = ROOT / cfg["sbp"]["outputs"]["sediment_tif"]
    mbes_path = ROOT / cfg["mbes"]["bathymetry_tif"]
    out_path = ROOT / cfg["viewer"]["sediment_rgb_tif"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    palette = np.array([hex_to_rgb(c) for c in SEDIMENT_COLORS], dtype=np.uint8)

    with rasterio.open(src_path) as src:
        cls = src.read(1)
        profile = src.profile.copy()

    with rasterio.open(mbes_path) as mb:
        dem = mb.read(1).astype(np.float32)
        if mb.nodata is not None:
            dem[dem == mb.nodata] = np.nan
    mbes_mask = np.isfinite(dem)

    if mbes_mask.shape != cls.shape:
        raise ValueError(
            f"MBES shape {mbes_mask.shape} != sediment shape {cls.shape}; "
            "regrid MBES to sediment grid first."
        )

    h, w = cls.shape
    # Initialize all pixels to NODATA_RGB; valid pixels overwritten below
    rgb = np.full((3, h, w), NODATA_RGB[0], dtype=np.uint8)
    for ch in range(3):
        rgb[ch].fill(NODATA_RGB[ch])

    for i, color in enumerate(palette):
        mask = (cls == i) & mbes_mask
        rgb[0][mask] = color[0]
        rgb[1][mask] = color[1]
        rgb[2][mask] = color[2]

    profile.update(
        count=3,
        dtype="uint8",
        nodata=NODATA_RGB[0],   # tile server uses this for transparency
        photometric="RGB",
        compress="lzw",
    )
    if "alpha" in profile:
        del profile["alpha"]

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(rgb[0], 1)
        dst.write(rgb[1], 2)
        dst.write(rgb[2], 3)

    n_visible = ((cls != -1) & mbes_mask).sum()
    print(f"Saved: {out_path.relative_to(ROOT)}")
    print(f"  size           : {h}x{w}")
    print(f"  classes        : {len(palette)}")
    print(f"  MBES coverage  : {100 * mbes_mask.sum() / (h * w):.1f}%")
    print(f"  visible pixels : {100 * n_visible / (h * w):.1f}%")
    print(f"  nodata RGB     : {NODATA_RGB} (transparent in viewer)")


if __name__ == "__main__":
    main()