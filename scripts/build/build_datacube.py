"""
Build Overviews (Pyramids) for all GeoTIFF layers.
This drastically improves WebGIS rendering performance (e.g., rio-tiler).
"""
from pathlib import Path
import rasterio
from rasterio.enums import Resampling

ROOT    = Path(__file__).parent.parent.parent
TIF_DIR = ROOT / "outputs/tif"

# 縮放倍率 (金字塔層級：1/2, 1/4, 1/8, 1/16, 1/32)
FACTORS = [2, 4, 8, 16, 32]


def main():
    if not TIF_DIR.exists():
        print(f"Error: Directory not found -> {TIF_DIR}")
        return

    tif_files = list(TIF_DIR.glob("*.tif"))
    if not tif_files:
        print(f"No TIFF files found in {TIF_DIR}")
        return

    print(f"Found {len(tif_files)} TIFF files. Building overviews...\n")

    success_count = 0
    for tif_path in tif_files:
        try:
            with rasterio.open(tif_path, 'r+') as dst:

                dst.build_overviews(FACTORS, Resampling.average)
                dst.update_tags(ns='rio_overview', resampling='average')
                
            print(f"  Processed: {tif_path.name}")
            success_count += 1
            
        except Exception as e:
            print(f"  FAILED: {tif_path.name} -> {e}")

    print(f"\n✅ Successfully generated overviews for {success_count}/{len(tif_files)} layers!")


if __name__ == "__main__":
    main()