# scripts/build/build_sediment.py
"""
Build sediment classification layer from RL tif.
Simple Hamilton table lookup - no RF, no JSF parsing.

Input:  sbp_rl.tif
Output: sbp_sediment_class.tif
"""
from pathlib import Path

import numpy as np
import rasterio

from src.sbp.calculation import SEDIMENT_LABELS, classify_sediment

ROOT    = Path(__file__).parent.parent.parent
RL_TIF  = ROOT / "outputs/tif/sbp_rl.tif"
OUT_SED = ROOT / "outputs/tif/sbp_sediment_class.tif"


def main():
    with rasterio.open(RL_TIF) as src:
        rl = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        rl[rl == nodata] = np.nan

    finite = np.isfinite(rl)
    print(f"RL pixels: {finite.sum()}")

    sed = np.full(rl.shape, -1, dtype=np.int8)
    sed[finite] = np.array([classify_sediment(v) for v in rl[finite].ravel()])

    # write
    out_profile = profile.copy()
    out_profile.update(dtype="int8", count=1, nodata=-1)
    with rasterio.open(OUT_SED, "w", **out_profile) as dst:
        dst.write(sed, 1)
    print(f"Saved: {OUT_SED}")

    # summary
    total = (sed >= 0).sum()
    print(f"\nSediment class distribution:")
    for i, label in enumerate(SEDIMENT_LABELS):
        count = (sed == i).sum()
        print(f"  {i} {label:20s}: {count:6d} px ({100 * count / max(total, 1):.1f}%)")


if __name__ == "__main__":
    main()