# scripts/build/build_isopach.py
"""
Build isopach (sediment thickness) layer from SBP data.

Workflow:
  1. Extract RL per ping to determine local Vp (Hamilton 1980)
  2. Track Sub-Bottom Reflector (SBR) using Peak Prominence
  3. Filter false multiples using sonar draft geometry
  4. Compute physical thickness via Two-Way Travel Time (TWT)
  5. Spatial interpolation (IDW) to full grid

Vp source: directly from per-ping RL, not from RF-predicted sediment class.
"""
import numpy as np
import rasterio
from pathlib import Path
from pyproj import Transformer
from scipy.signal import find_peaks, hilbert
from scipy.spatial import cKDTree
from tqdm import tqdm

from src.data_loader.read_sbp_jsf import read_sbp_jsf
from src.sbp.calculation import compute_rl_batch
from src.config import EPSG, SOUND_SPEED, SBP_CC

ROOT      = Path(__file__).parent.parent.parent
SBP_PATH  = ROOT / "data/sbp"
MBES_TIF  = ROOT / "outputs/tif/mbes_bathymetry.tif"
OUT_THICK = ROOT / "outputs/tif/sbp_isopach.tif"

NS          = 20480e-9
THICK_CLIP  = (0.1, 3.0)
SONAR_DRAFT = 1.04

MAX_GAP     = 70
K_NEIGHBORS = 200
SMOOTHING   = MAX_GAP * 2


def rl_to_vp(rl_db):
    """Map per-ping RL to Vp using Hamilton (1980) ratios."""
    if not np.isfinite(rl_db) or rl_db < 0:
        return SOUND_SPEED
    if rl_db < 7.33:
        return SOUND_SPEED * 1.201   # Coarse sand
    elif rl_db < 8.02:
        return SOUND_SPEED * 1.145   # Fine sand
    elif rl_db < 8.73:
        return SOUND_SPEED * 1.115   # Very fine sand
    elif rl_db < 9.63:
        return SOUND_SPEED * 1.078   # Silty sand
    elif rl_db < 9.82:
        return SOUND_SPEED * 1.080   # Sandy silt
    elif rl_db < 10.25:
        return SOUND_SPEED * 1.057   # Silt
    elif rl_db < 11.98:
        return SOUND_SPEED * 1.033   # Sandy-silt-clay
    elif rl_db < 13.20:
        return SOUND_SPEED * 1.014   # Silty clay
    elif rl_db < 13.37:
        return SOUND_SPEED * 0.994   # Clayey silt
    elif rl_db < 23.95:
        return SOUND_SPEED * 0.98   # Compacted mud
    else:
        return SOUND_SPEED * 0.9     # Fluid mud


def find_b(amps, blanking=50):
    """Locate primary seafloor return."""
    return int(np.argmax(amps[blanking:])) + blanking


def extract_valid_thickness(amps, idx_b, local_v):
    """
    Extract sub-bottom reflector and compute sediment thickness.
    Uses local Vp for both blind-zone and depth calculations.
    """
    local_sdep = NS * local_v / 2

    s = idx_b + int(0.15 / local_sdep)
    e = min(len(amps), idx_b + int(4.0 / local_sdep))

    if e - s < 10:
        return None

    window   = amps[s:e].astype(np.float64)
    envelope = np.abs(hilbert(window))
    db_win   = 20 * np.log10(np.maximum(envelope, 1.0))
    noise_floor = np.median(db_win) + 5.0

    peaks, properties = find_peaks(
        db_win,
        prominence=3.0,
        distance=int(0.15 / local_sdep),
        height=noise_floor,
        width=2,
    )

    if len(peaks) == 0:
        return None

    best_idx     = np.argmax(properties["prominences"] * properties["peak_heights"])
    best_peak_pos = peaks[best_idx]
    thick        = (s + best_peak_pos - idx_b) * local_sdep

    # reject sonar draft multiple
    if abs(thick - SONAR_DRAFT) < 0.15:
        return None

    return thick


def main():
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)

    # ── 1. Extract thickness along tracklines ────────────────
    print("1. Extracting acoustic thickness from SBP pings...")
    all_x, all_y, all_thick = [], [], []
    n_total = n_valid = n_rejected = 0

    for jsf in tqdm(sorted(SBP_PATH.glob("*.jsf")), desc="Processing JSF"):
        data = read_sbp_jsf(jsf)
        if "SBP" not in data:
            continue
        d     = data["SBP"]
        valid = ~np.isnan(d["lon"])
        if not valid.any():
            continue

        x, y = transformer.transform(d["lon"][valid], d["lat"][valid])
        amps  = d["amps"][valid].astype(np.float32)

        # vectorized RL for all pings → local Vp
        rls = compute_rl_batch(amps, SBP_CC)

        for i in range(len(amps)):
            n_total += 1
            local_v = rl_to_vp(rls[i])
            idx_b   = find_b(amps[i])
            thick   = extract_valid_thickness(amps[i], idx_b, local_v)

            if thick is not None and THICK_CLIP[0] <= thick <= THICK_CLIP[1]:
                all_x.append(x[i])
                all_y.append(y[i])
                all_thick.append(thick)
                n_valid += 1
            else:
                n_rejected += 1

    all_x     = np.array(all_x,     dtype=np.float64)
    all_y     = np.array(all_y,     dtype=np.float64)
    all_thick = np.array(all_thick, dtype=np.float64)

    print(f"\nExtraction summary:")
    print(f"  Total pings       : {n_total}")
    print(f"  Valid SBR detected: {n_valid} ({100*n_valid/n_total:.1f}%)")
    if len(all_thick):
        print(f"  Thickness range   : {all_thick.min():.2f} ~ {all_thick.max():.2f} m")
        print(f"  Thickness median  : {np.median(all_thick):.2f} m")

    if n_valid < 100:
        print("Error: not enough valid points for interpolation.")
        return

    # ── 2. Load MBES grid geometry ───────────────────────────
    print("\n2. Loading MBES grid geometry...")
    with rasterio.open(MBES_TIF) as src:
        profile     = src.profile.copy()
        transform   = src.transform
        height, width = src.height, src.width
        mbes_data   = src.read(1)
        mbes_nodata = src.nodata

    res     = transform.a
    xs_grid = transform.c + (np.arange(width)  + 0.5) * res
    ys_grid = transform.f + (np.arange(height) + 0.5) * (-res)
    gx, gy  = np.meshgrid(xs_grid, ys_grid)

    valid_grid = (
        (mbes_data != mbes_nodata).ravel()
        if mbes_nodata is not None
        else np.isfinite(mbes_data).ravel()
    )

    target_x = gx.ravel()[valid_grid]
    target_y = gy.ravel()[valid_grid]

    # ── 3. IDW interpolation ──────────────────────────────────
    print("\n3. IDW interpolation...")
    points       = np.column_stack([all_x, all_y])
    tree         = cKDTree(points)
    target_pts   = np.column_stack([target_x, target_y])
    max_distance = MAX_GAP * 1.5

    distances, indices = tree.query(
        target_pts, k=K_NEIGHBORS, distance_upper_bound=max_distance
    )

    interp_thick = np.full(len(target_pts), np.nan, dtype=np.float32)

    for i in range(len(target_pts)):
        valid_k = np.isfinite(distances[i]) & (distances[i] <= max_distance)
        if not valid_k.any():
            continue
        d_k = distances[i][valid_k]
        v_k = all_thick[indices[i][valid_k]]
        w   = 1.0 / (d_k ** 2 + SMOOTHING)
        interp_thick[i] = np.sum(w * v_k) / np.sum(w)

    thick_2d = np.full((height, width), -9999.0, dtype=np.float32)
    valid_interp = np.isfinite(interp_thick)
    flat_idx = np.where(valid_grid)[0]
    thick_2d.ravel()[flat_idx[valid_interp]] = interp_thick[valid_interp]

    # ── 4. Write GeoTIFF ──────────────────────────────────────
    print("\n4. Saving isopach GeoTIFF...")
    profile.update(dtype="float32", count=1, nodata=-9999.0)
    with rasterio.open(OUT_THICK, "w", **profile) as dst:
        dst.write(thick_2d, 1)

    valid_t = thick_2d[thick_2d != -9999.0]
    print(f"Saved: {OUT_THICK}")
    print(f"Thickness range  : {valid_t.min():.2f} ~ {valid_t.max():.2f} m")
    print(f"Thickness median : {np.median(valid_t):.2f} m")
    print(f"Coverage         : {len(valid_t)} / {height*width} px "
          f"({100*len(valid_t)/(height*width):.1f}%)")


if __name__ == "__main__":
    main()