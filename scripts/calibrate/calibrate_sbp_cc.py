# scripts/calibrate/calibrate_sbp_cc.py
"""
SBP Calibration Coefficient (CC) estimation.

Automatically finds flat, uniform-substrate segments from SBP data,
computes per-ping CC values, and selects the most stable segment
to derive the final CC constant.

Output: prints the recommended CC value to set in src/config.py

Methodology:
  1. For each JSF file, sample VRM and depth at ping locations
  2. Find consecutive runs of pings where VRM < threshold and
     depth variation is small (flat, uniform bottom)
  3. Compute per-ping CC using bottom echo (B) and
     bottom-surface-bottom echo (BSB) ratio (Huang & Liu, 2015, Eq.6)
  4. Rank segments by CC stability (coefficient of variation)
  5. Select top segments and compute combined CC

Reference:
  Huang & Liu (2015) Eq.(6): CC = (r1*A1)^2 / (r2*A2)
"""
from pathlib import Path
import numpy as np
import rasterio
from pyproj import Transformer
from src.data_loader.read_sbp_jsf import read_sbp_jsf
from src.sbp.calculation import estimate_cc

ROOT     = Path(__file__).parent.parent.parent
SBP_PATH = ROOT / "data/sbp"
MBES_TIF = ROOT / "outputs/tif/mbes_bathymetry.tif"
VRM_TIF  = ROOT / "outputs/tif/mbes_vrm.tif"

# selection criteria
MIN_CONSEC      = 30    # minimum consecutive good pings
VRM_THRESH      = 0.002 # maximum VRM (low = flat)
DEPTH_STD_THRESH = 0.5  # maximum depth std within segment (m)
MIN_CC_COUNT    = 10    # minimum valid CC estimates per segment
TOP_N_SEGMENTS  = 3     # number of best segments to combine


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
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True)
    jsf_files = sorted(SBP_PATH.glob("*.jsf"))

    print("=" * 70)
    print("SBP Calibration Coefficient (CC) Estimation")
    print("=" * 70)
    print(f"\nCriteria:")
    print(f"  VRM            < {VRM_THRESH}")
    print(f"  Depth std      < {DEPTH_STD_THRESH} m")
    print(f"  Min consec.    >= {MIN_CONSEC} pings")
    print(f"  Min CC samples >= {MIN_CC_COUNT}")
    print(f"  JSF files      : {len(jsf_files)}")
    print()

    # scan all files for candidate segments
    candidates = []

    for jsf in jsf_files:
        data = read_sbp_jsf(jsf)
        if "SBP" not in data:
            continue
        d = data["SBP"]
        valid = ~np.isnan(d["lon"])
        if valid.sum() < MIN_CONSEC:
            continue

        x, y = tr.transform(d["lon"][valid], d["lat"][valid])
        vrm = sample_raster(VRM_TIF, x, y)
        depth = sample_raster(MBES_TIF, x, y)

        good = np.isfinite(vrm) & (vrm < VRM_THRESH) & np.isfinite(depth)
        start, length = find_longest_run(good)

        if length < MIN_CONSEC:
            continue

        seg_depth = depth[start:start + length]
        seg_vrm = vrm[start:start + length]
        depth_std = np.nanstd(seg_depth)

        if depth_std > DEPTH_STD_THRESH:
            continue

        # compute CC for this segment
        amps_valid = d["amps"][valid]
        ccs = []
        for i in range(start, start + length):
            cc = estimate_cc(amps_valid[i])
            if cc is not None and cc > 0:
                ccs.append(cc)

        if len(ccs) < MIN_CC_COUNT:
            continue

        ccs = np.array(ccs)
        # IQR outlier removal
        q25, q75 = np.percentile(ccs, 25), np.percentile(ccs, 75)
        iqr = q75 - q25
        clean = ccs[(ccs >= q25 - 1.5 * iqr) & (ccs <= q75 + 1.5 * iqr)]

        if len(clean) < MIN_CC_COUNT:
            continue

        cv = clean.std() / clean.mean() * 100

        candidates.append({
            "file": jsf.name,
            "start": start,
            "length": length,
            "depth_mean": float(np.nanmean(seg_depth)),
            "depth_std": depth_std,
            "vrm_mean": float(np.nanmean(seg_vrm)),
            "cc_median": float(np.median(clean)),
            "cc_cv": cv,
            "cc_count": len(clean),
            "cc_values": clean,
        })

    if not candidates:
        print("ERROR: No valid calibration segments found.")
        print("Try relaxing criteria (increase VRM_THRESH or DEPTH_STD_THRESH).")
        return

    # sort by CV (most stable first)
    candidates.sort(key=lambda c: c["cc_cv"])

    # print all candidates
    print(f"Found {len(candidates)} candidate segments (sorted by stability):\n")
    print(f"{'#':>2s} {'File':<28s} {'Ping':>5s} {'Len':>4s} "
          f"{'Depth':>6s} {'D_std':>5s} {'VRM':>8s} "
          f"{'CC_dB':>7s} {'CV%':>6s} {'N':>4s}")
    print("-" * 90)

    for i, c in enumerate(candidates):
        marker = " <--" if i < TOP_N_SEGMENTS else ""
        print(f"{i+1:2d} {c['file']:<28s} {c['start']:5d} {c['length']:4d} "
              f"{c['depth_mean']:6.1f} {c['depth_std']:5.2f} "
              f"{c['vrm_mean']:8.5f} "
              f"{20*np.log10(c['cc_median']):7.1f} {c['cc_cv']:6.1f} "
              f"{c['cc_count']:4d}{marker}")

    # combine top segments
    # collect all CC_dB values
    all_cc_db = np.array([20 * np.log10(c["cc_median"]) for c in candidates])

    # find the dominant cluster using median + MAD
    median_db = np.median(all_cc_db)
    mad = np.median(np.abs(all_cc_db - median_db))
    threshold = 3.0 * mad  # ~2 sigma equivalent

    consistent = [c for c, db in zip(candidates, all_cc_db)
                  if abs(db - median_db) <= max(threshold, 3.0)]

    if not consistent:
        print("WARNING: No consistent segments found, using all.")
        consistent = candidates

    # from consistent segments, pick top N by CV
    consistent.sort(key=lambda c: c["cc_cv"])
    top = consistent[:TOP_N_SEGMENTS]

    print(f"\nCC clustering: median={median_db:.1f} dB, MAD={mad:.1f} dB, "
          f"threshold=±{max(threshold, 3.0):.1f} dB")
    print(f"Consistent segments: {len(consistent)} / {len(candidates)}")

    all_cc = np.concatenate([c["cc_values"] for c in top])
    final_cc = float(np.median(all_cc))
    final_cc_db = 20 * np.log10(final_cc)
    combined_cv = all_cc.std() / all_cc.mean() * 100

    print(f"\n{'=' * 70}")
    print(f"RESULT: Combined CC from top {TOP_N_SEGMENTS} segments")
    print(f"{'=' * 70}")
    print(f"  CC         = {final_cc:.6e}")
    print(f"  CC (dB)    = {final_cc_db:.2f} dB")
    print(f"  Combined N = {len(all_cc)}")
    print(f"  Combined CV= {combined_cv:.1f}%")
    print(f"\n  Sources:")
    for c in top:
        print(f"    {c['file']} (ping {c['start']}-{c['start']+c['length']}, "
              f"depth={c['depth_mean']:.1f}m, CV={c['cc_cv']:.1f}%)")

    print(f"\n  Add to src/config.py:")
    print(f"  SBP_CC = {final_cc:.6e}")
    print()


if __name__ == "__main__":
    main()