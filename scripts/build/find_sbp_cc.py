"""
SBP Calibration Coefficient (CC) estimation.

Selects multiple stable segments per file using terrain-derived flatness
indicators (VRM, slope, BPI), depth uniformity, and within-segment
substrate uniformity (amp_b CV). Computes one CC per segment via
ping-averaging (Huang & Liu 2015), then takes the median CC across
all qualifying segments as the survey-wide calibration constant.

Writes the result to stdout; user should update
configs/mudan.yaml sbp.calibration_constant.
"""
import numpy as np
import rasterio
from pyproj import Transformer

from src.config import get_config, ROOT
from src.sbp.read_sbp_jsf import read_sbp_jsf
from src.sbp.calculation import estimate_cc
from src.sbp.config import (
    BLANKING_SAMPLES,
    CC_MIN_CONSEC_PINGS, CC_VRM_PERCENTILE,
    CC_SLOPE_MAX_DEG, CC_BPI_ABS_MAX,
    CC_DEPTH_STD_MAX, CC_AMP_CV_MAX,
    CC_MIN_START_PING,
)


def sample_raster(tif_path, xs, ys):
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        nd, t = src.nodata, src.transform
        h, w = data.shape
    cols = ((xs - t.c) / t.a).astype(int)
    rows = ((ys - t.f) / t.e).astype(int)
    v = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    vals = np.full(len(xs), np.nan)
    vals[v] = data[rows[v], cols[v]]
    if nd is not None:
        vals[vals == nd] = np.nan
    return vals


def find_all_runs(bool_arr, min_length, min_start=0):
    """Yield (start, length) for every run of True values >= min_length.
    Skips runs starting before min_start (warm-up exclusion)."""
    runs = []
    cur_s, cur_n = 0, 0
    for i, v in enumerate(bool_arr):
        if v:
            if cur_n == 0:
                cur_s = i
            cur_n += 1
        else:
            if cur_n >= min_length and cur_s >= min_start:
                runs.append((cur_s, cur_n))
            cur_n = 0
    # tail run
    if cur_n >= min_length and cur_s >= min_start:
        runs.append((cur_s, cur_n))
    return runs


def main():
    cfg = get_config()
    epsg = cfg["grid"]["epsg"]
    mbes_tif = ROOT / cfg["mbes"]["bathymetry_tif"]
    vrm_tif = ROOT / cfg["mbes"]["vrm_tif"]
    slope_tif = ROOT / cfg["mbes"]["slope_tif"]
    bpi_tif = ROOT / cfg["mbes"]["bpi_tif"]
    sbp_dirs = [ROOT / d["path"] for d in cfg["sbp"]["survey_dirs"]]

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    jsf_files = []
    for d in sbp_dirs:
        jsf_files.extend(sorted(d.glob("*.jsf")))

    print("=" * 70)
    print("SBP Calibration Coefficient (CC) Estimation")
    print("=" * 70)
    print(f"JSF files       : {len(jsf_files)}")
    print(f"Method          : Segment averaging")
    print(f"Selection criteria:")
    print(f"  VRM percentile     < {CC_VRM_PERCENTILE}")
    print(f"  Slope (deg)        < {CC_SLOPE_MAX_DEG}")
    print(f"  |BPI| (m)          < {CC_BPI_ABS_MAX}")
    print(f"  Depth std (m)      < {CC_DEPTH_STD_MAX}")
    print(f"  amp_b CV           < {CC_AMP_CV_MAX}")
    print(f"  Min consec pings  >= {CC_MIN_CONSEC_PINGS}")
    print(f"  Min start ping    >= {CC_MIN_START_PING}")
    print()

    # ── Pool VRM threshold ──
    print("Computing VRM threshold from full survey...")
    all_vrm = []
    for jsf in jsf_files:
        data = read_sbp_jsf(jsf)
        if "SBP" not in data:
            continue
        d = data["SBP"]
        valid = ~np.isnan(d["lon"])
        if valid.sum() == 0:
            continue
        x, y = transformer.transform(d["lon"][valid], d["lat"][valid])
        vrm = sample_raster(vrm_tif, x, y)
        all_vrm.extend(vrm[np.isfinite(vrm)].tolist())

    if not all_vrm:
        print("ERROR: No VRM data.")
        return
    vrm_thresh = float(np.percentile(all_vrm, CC_VRM_PERCENTILE))
    print(f"VRM threshold    : {vrm_thresh:.5f}\n")

    # ── Find all qualifying segments ──
    candidates = []
    for jsf in jsf_files:
        data = read_sbp_jsf(jsf)
        if "SBP" not in data:
            continue
        d = data["SBP"]
        valid = ~np.isnan(d["lon"])
        if valid.sum() < CC_MIN_CONSEC_PINGS:
            continue

        x, y = transformer.transform(d["lon"][valid], d["lat"][valid])
        vrm = sample_raster(vrm_tif, x, y)
        slope = sample_raster(slope_tif, x, y)
        bpi = sample_raster(bpi_tif, x, y)
        depth = sample_raster(mbes_tif, x, y)

        good = (
            np.isfinite(vrm) & (vrm < vrm_thresh)
            & np.isfinite(slope) & (slope < CC_SLOPE_MAX_DEG)
            & np.isfinite(bpi) & (np.abs(bpi) < CC_BPI_ABS_MAX)
            & np.isfinite(depth)
        )

        amps_valid = d["amps"][valid]
        runs = find_all_runs(good, CC_MIN_CONSEC_PINGS, CC_MIN_START_PING)

        for start, length in runs:
            seg_depth = depth[start:start + length]
            if np.nanstd(seg_depth) > CC_DEPTH_STD_MAX:
                continue

            amps_seg = amps_valid[start:start + length]
            search = amps_seg[:, BLANKING_SAMPLES:]
            idx_b = np.argmax(search, axis=1) + BLANKING_SAMPLES
            amp_b = amps_seg[np.arange(len(amps_seg)), idx_b]
            amp_b_pos = amp_b[amp_b > 0]
            if len(amp_b_pos) < CC_MIN_CONSEC_PINGS // 2:
                continue

            amp_cv = float(np.std(amp_b_pos) / np.mean(amp_b_pos))
            if amp_cv > CC_AMP_CV_MAX:
                continue

            cc = estimate_cc(amps_seg)
            if cc is None or cc <= 0:
                continue

            candidates.append({
                "file": jsf.name,
                "start": start,
                "length": length,
                "depth_mean": float(np.nanmean(seg_depth)),
                "depth_std": float(np.nanstd(seg_depth)),
                "amp_cv": amp_cv,
                "cc_db": 20 * np.log10(cc),
                "cc": cc,
            })

    if not candidates:
        print("ERROR: No qualifying segments found. Relax criteria.")
        return

    # ── Report ──
    candidates.sort(key=lambda c: c["cc_db"])
    print(f"Found {len(candidates)} qualifying segments:\n")
    print(f"{'#':>3s} {'File':<28s} {'Ping':>5s} {'Len':>4s} "
          f"{'Depth':>6s} {'D_std':>5s} {'amp_CV':>7s} {'CC_dB':>7s}")
    print("-" * 76)
    for i, c in enumerate(candidates):
        print(f"{i+1:3d} {c['file']:<28s} {c['start']:5d} {c['length']:4d} "
              f"{c['depth_mean']:6.1f} {c['depth_std']:5.2f} "
              f"{c['amp_cv']:7.3f} {c['cc_db']:7.1f}")

    # ── Aggregate ──
    cc_db_arr = np.array([c["cc_db"] for c in candidates])
    final_cc_db = float(np.median(cc_db_arr))
    final_cc = 10 ** (final_cc_db / 20)
    cc_std = float(np.std(cc_db_arr))

    # Bootstrap median CI
    if len(cc_db_arr) >= 3:
        rng = np.random.default_rng(42)
        idx = rng.integers(0, len(cc_db_arr), size=(2000, len(cc_db_arr)))
        boot_med = np.median(cc_db_arr[idx], axis=1)
        ci_lo = float(np.percentile(boot_med, 2.5))
        ci_hi = float(np.percentile(boot_med, 97.5))
    else:
        ci_lo = ci_hi = float("nan")

    print(f"\n{'=' * 70}")
    print(f"RESULT")
    print(f"{'=' * 70}")
    print(f"  Segments         : {len(candidates)}")
    print(f"  CC median        : {final_cc_db:.2f} dB")
    print(f"  CC std           : {cc_std:.2f} dB")
    print(f"  CC 95% CI        : [{ci_lo:.2f}, {ci_hi:.2f}] dB "
          f"(width {ci_hi - ci_lo:.2f} dB)")
    print(f"  CC linear        : {final_cc:.4e}")
    print()
    print(f"Update configs/mudan.yaml sbp.calibration_constant:")
    print(f"  calibration_constant: {final_cc:.4e}")


if __name__ == "__main__":
    main()