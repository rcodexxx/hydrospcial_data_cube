"""
Diagnostic: incident angle SNR / signal stability for SSS.

Pools all georef'd samples across the survey, bins by incident angle,
reports per-bin: median dB, robust CV, outlier ratio, sample count.

Important caveat:
  georef_line internally clips slant range using:
    min_slant = altitude / cos(15 deg)
    max_slant = altitude / cos(70 deg)
  so samples above 70 deg are NOT in the pool. To probe 70-80 deg,
  temporarily widen that clip in georef.py, then re-run.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from tqdm import tqdm

from src.config import get_config, ROOT
from src.sss.georef import georef_line


# ── parameters (hardcoded; one-off diagnostic) ────────────────
ANGLE_MIN = 15.0
ANGLE_MAX = 80.0
ANGLE_STEP = 1.0
OUTLIER_Z = 3.0
MIN_SAMPLES_PER_BIN = 100
OUT_DIR = ROOT / "outputs" / "diagnose"

# Match read_sss_jsf channel keys
CHANNELS = ["HF_port", "HF_stbd", "LF_port", "LF_stbd"]


def collect_samples(jsf_files, mbes_tif, channel, mbes):
    """Pool (amp_db, inc_angle) across all lines for one channel."""
    amps_db, angles = [], []
    n_skip = 0
    for jsf_path, cable in tqdm(jsf_files, desc=f"  {channel}"):
        r = georef_line(jsf_path, mbes_tif, channel,
                        cable_length=cable, mbes_preloaded=mbes)
        if r is None:
            n_skip += 1
            continue
        bs = r["bs_linear"]
        ang = r["inc_angle"]
        valid = (bs > 0) & np.isfinite(ang)
        if not valid.any():
            continue
        amps_db.append(10.0 * np.log10(bs[valid]))
        angles.append(ang[valid])

    if not amps_db:
        print(f"    no valid data on any line ({n_skip}/{len(jsf_files)} skipped)")
        return None, None
    print(f"    skipped {n_skip}/{len(jsf_files)} lines")
    return np.concatenate(amps_db), np.concatenate(angles)


def per_bin_stats(amp_db, angle):
    bins = np.arange(ANGLE_MIN, ANGLE_MAX + ANGLE_STEP, ANGLE_STEP)
    centers = (bins[:-1] + bins[1:]) / 2

    n_bins = len(centers)
    median = np.full(n_bins, np.nan)
    cv = np.full(n_bins, np.nan)
    outlier_ratio = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)

    bin_idx = np.digitize(angle, bins) - 1
    for j in range(n_bins):
        sel = amp_db[bin_idx == j]
        count[j] = len(sel)
        if count[j] < MIN_SAMPLES_PER_BIN:
            continue
        med = np.median(sel)
        mad = np.median(np.abs(sel - med))
        median[j] = med
        cv[j] = (1.4826 * mad) / max(abs(med), 1e-6)
        z = (sel - med) / max(1.4826 * mad, 1e-6)
        outlier_ratio[j] = float(np.mean(np.abs(z) > OUTLIER_Z))

    return centers, median, cv, outlier_ratio, count


def plot(centers, median, cv, outlier, count, title, out_path):
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(centers, median, "-o", ms=3)
    axes[0].set_ylabel("Median BS (dB)")
    axes[0].grid(alpha=0.3)
    axes[0].axvline(70, color="red", ls="--", lw=1, label="current cutoff")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(centers, cv, "-o", ms=3, color="darkorange")
    axes[1].set_ylabel("Robust CV\n(MAD/|median|)")
    axes[1].grid(alpha=0.3)
    axes[1].axvline(70, color="red", ls="--", lw=1)

    axes[2].plot(centers, outlier * 100, "-o", ms=3, color="purple")
    axes[2].set_ylabel("Outlier ratio\n(|z|>3, %)")
    axes[2].grid(alpha=0.3)
    axes[2].axvline(70, color="red", ls="--", lw=1)

    axes[3].semilogy(centers, np.maximum(count, 1), "-o", ms=3, color="gray")
    axes[3].set_ylabel("Sample count")
    axes[3].set_xlabel("Incident angle (deg)")
    axes[3].grid(alpha=0.3, which="both")
    axes[3].axvline(70, color="red", ls="--", lw=1)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"    saved: {out_path.name}")


def main():
    cfg = get_config()
    sss_cfg = cfg["sss"]
    mbes_tif = ROOT / cfg["mbes"]["bathymetry_tif"]

    # Preload MBES once (was being read fresh for every line before)
    print("Preloading MBES grid...")
    with rasterio.open(mbes_tif) as src:
        mbes = {
            "data": src.read(1).astype(np.float32),
            "transform": src.transform,
            "tr": Transformer.from_crs(
                "EPSG:4326", f"EPSG:{src.crs.to_epsg()}", always_xy=True),
        }

    # Correct config key: "survey_dirs", not "surveys"
    jsf_files = []
    for entry in sss_cfg["survey_dirs"]:
        cable = entry.get("cable_length")
        for f in sorted((ROOT / entry["path"]).glob("*.jsf")):
            jsf_files.append((f, cable))
    print(f"Total JSF files: {len(jsf_files)}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for ch in CHANNELS:
        print(f"Channel: {ch}")
        amp_db, angle = collect_samples(jsf_files, mbes_tif, ch, mbes)
        if amp_db is None:
            print()
            continue
        print(f"    pooled samples: {len(amp_db):,}")
        centers, median, cv, outlier, count = per_bin_stats(amp_db, angle)
        plot(centers, median, cv, outlier, count,
             f"Angle SNR diagnostic - {ch}",
             OUT_DIR / f"angle_snr_{ch}.png")

        tail = (centers >= 60) & (centers <= 80)
        for i in np.where(tail)[0]:
            summary.append((ch, centers[i], count[i], median[i],
                            cv[i], outlier[i]))
        print()

    print("=== Tail summary (60-80 deg) ===")
    print(f"{'channel':<10} {'angle':>6} {'count':>10} "
          f"{'median_dB':>10} {'CV':>8} {'outlier%':>9}")
    for ch, ang, n, med, c, o in summary:
        med_s = f"{med:.2f}" if np.isfinite(med) else "  nan"
        c_s = f"{c:.3f}" if np.isfinite(c) else "  nan"
        o_s = f"{o*100:.1f}" if np.isfinite(o) else " nan"
        print(f"{ch:<10} {ang:>6.1f} {n:>10,} {med_s:>10} {c_s:>8} {o_s:>9}")


if __name__ == "__main__":
    main()