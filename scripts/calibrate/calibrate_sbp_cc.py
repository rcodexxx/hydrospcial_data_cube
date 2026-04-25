"""
SBP Calibration Coefficient (CC) estimation.

Finds flat, uniform-substrate segments from SBP data, computes one CC
per segment via ping-averaging (Huang & Liu 2015 method), and selects
the cluster of consistent segments to derive the final CC.

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
    CC_MIN_CONSEC_PINGS, CC_VRM_PERCENTILE, CC_DEPTH_STD_MAX,
    CC_TOP_N_SEGMENTS,
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


def find_longest_run(bool_arr):
    best_s, best_n = 0, 0
    cur_s, cur_n = 0, 0
    for i, v in enumerate(bool_arr):
        if v:
            cur_n += 1
            if cur_n > best_n:
                best_n, best_s = cur_n, cur_s
        else:
            cur_s, cur_n = i + 1, 0
    return best_s, best_n


def main():
    cfg = get_config()
    epsg = cfg["grid"]["epsg"]
    mbes_tif = ROOT / cfg["mbes"]["bathymetry_tif"]
    vrm_tif = ROOT / cfg["mbes"]["vrm_tif"]
    sbp_dirs = [ROOT / d["path"] for d in cfg["sbp"]["survey_dirs"]]

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    jsf_files = []
    for d in sbp_dirs:
        jsf_files.extend(sorted(d.glob("*.jsf")))

    print("=" * 70)
    print("SBP Calibration Coefficient (CC) Estimation")
    print("=" * 70)
    print(f"JSF files      : {len(jsf_files)}")
    print(f"Method         : Segment averaging (Huang & Liu 2015)")
    print(f"Criteria:")
    print(f"  VRM percentile < {CC_VRM_PERCENTILE}")
    print(f"  Depth std      < {CC_DEPTH_STD_MAX} m")
    print(f"  Min consec     >= {CC_MIN_CONSEC_PINGS} pings")
    print()

    # Pool all VRM to determine threshold
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
        print("ERROR: No valid VRM data found.")
        return

    vrm_thresh = float(np.percentile(all_vrm, CC_VRM_PERCENTILE))
    print(f"VRM threshold  : {vrm_thresh:.5f} (percentile {CC_VRM_PERCENTILE})\n")

    # Find candidate segments and compute one CC per segment
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
        depth = sample_raster(mbes_tif, x, y)

        good = np.isfinite(vrm) & (vrm < vrm_thresh) & np.isfinite(depth)
        start, length = find_longest_run(good)
        if length < CC_MIN_CONSEC_PINGS:
            continue

        seg_depth = depth[start:start + length]
        seg_vrm = vrm[start:start + length]
        depth_std = np.nanstd(seg_depth)
        if depth_std > CC_DEPTH_STD_MAX:
            continue

        amps_valid = d["amps"][valid]
        amps_segment = amps_valid[start:start + length]
        cc = estimate_cc(amps_segment)
        if cc is None or cc <= 0:
            continue

        candidates.append({
            "file": jsf.name,
            "start": start,
            "length": length,
            "depth_mean": float(np.nanmean(seg_depth)),
            "depth_std": depth_std,
            "vrm_mean": float(np.nanmean(seg_vrm)),
            "cc": cc,
            "cc_db": 20 * np.log10(cc),
        })

    if not candidates:
        print("ERROR: No valid calibration segments found.")
        return

    # Sort by stability (depth_std, then VRM)
    candidates.sort(key=lambda c: (c["depth_std"], c["vrm_mean"]))

    print(f"Found {len(candidates)} candidate segments "
          f"(sorted by depth stability):\n")
    print(f"{'#':>2s} {'File':<28s} {'Ping':>5s} {'Len':>4s} "
          f"{'Depth':>6s} {'D_std':>5s} {'VRM':>8s} {'CC_dB':>7s}")
    print("-" * 76)
    for i, c in enumerate(candidates):
        marker = " <--" if i < CC_TOP_N_SEGMENTS else ""
        print(f"{i + 1:2d} {c['file']:<28s} {c['start']:5d} {c['length']:4d} "
              f"{c['depth_mean']:6.1f} {c['depth_std']:5.2f} "
              f"{c['vrm_mean']:8.5f} {c['cc_db']:7.1f}{marker}")

    # Cluster consistency check
    all_cc_db = np.array([c["cc_db"] for c in candidates])
    median_db = np.median(all_cc_db)
    mad = np.median(np.abs(all_cc_db - median_db))
    threshold = max(3.0 * mad, 3.0)

    consistent = [
        c for c in candidates if abs(c["cc_db"] - median_db) <= threshold
    ]
    if not consistent:
        print("WARNING: No consistent segments, using all.")
        consistent = candidates

    consistent.sort(key=lambda c: (c["depth_std"], c["vrm_mean"]))
    top = consistent[:CC_TOP_N_SEGMENTS]

    print(f"\nCC clustering: median={median_db:.1f} dB, MAD={mad:.1f} dB, "
          f"threshold=±{threshold:.1f} dB")
    print(f"Consistent segments: {len(consistent)} / {len(candidates)}")

    top_cc_db = np.array([c["cc_db"] for c in top])
    final_cc_db = float(np.median(top_cc_db))
    final_cc = 10 ** (final_cc_db / 20)
    inter_seg_std = float(np.std(top_cc_db))

    print(f"\n{'=' * 70}")
    print(f"RESULT: CC from top {len(top)} most stable segments")
    print(f"{'=' * 70}")
    print(f"  CC              = {final_cc:.6e}")
    print(f"  CC (dB)         = {final_cc_db:.2f} dB")
    print(f"  Inter-segment σ = {inter_seg_std:.2f} dB")
    print(f"\n  Sources:")
    for c in top:
        print(f"    {c['file']} (ping {c['start']}-{c['start'] + c['length']}, "
              f"depth={c['depth_mean']:.1f}m, CC={c['cc_db']:.1f} dB)")

    print(f"\n  Update configs/mudan.yaml sbp.calibration_constant:")
    print(f"  calibration_constant: {final_cc:.6e}")


if __name__ == "__main__":
    main()