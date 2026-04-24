# scripts/verify/verify_arc.py
"""
ARC verification: plot before/after correction curves from saved npz files.

Usage:
    python scripts/verify/verify_arc.py --freq hf
    python scripts/verify/verify_arc.py --freq lf
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import CFG, ROOT

BINS = np.arange(15, 66, 1)
BIN_CENTERS = (BINS[:-1] + BINS[1:]) / 2


def build_arc_curve(bs, inc_angle):
    curve = np.full(len(BINS) - 1, np.nan)
    for j in range(len(BINS) - 1):
        mask = (inc_angle >= BINS[j]) & (inc_angle < BINS[j + 1]) & np.isfinite(bs)
        if mask.sum() >= 10:
            curve[j] = np.median(bs[mask])
    return curve


def arc_rms(curve):
    valid = np.isfinite(curve)
    if valid.sum() < 3:
        return np.nan
    return float(np.sqrt(np.nanmean((curve[valid] - np.nanmean(curve[valid])) ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", choices=["hf", "lf"], default="hf")
    args = parser.parse_args()

    corrected_dir = ROOT / CFG[f"sss_{args.freq}"]["outputs"]["corrected_dir"]
    npz_files = sorted(corrected_dir.glob("*.npz"))

    if not npz_files:
        print(f"No npz files found in {corrected_dir}")
        return

    print(f"Loading {len(npz_files)} files...")
    raw_bs_all,  raw_inc_all  = [], []
    corr_bs_all, corr_inc_all = [], []

    for npz_path in npz_files:
        d = np.load(npz_path)
        for side in ("port", "stbd"):
            if f"{side}_bs_corr" not in d:
                continue
            inc = d[f"{side}_inc"]
            n   = len(inc)
            idx = np.random.choice(n, min(n, 5000), replace=False)

            raw_bs_all.append(d[f"{side}_bs_raw"][idx])
            raw_inc_all.append(inc[idx])
            corr_bs_all.append(d[f"{side}_bs_corr"][idx])
            corr_inc_all.append(inc[idx])

    raw_bs   = np.concatenate(raw_bs_all)
    raw_inc  = np.concatenate(raw_inc_all)
    corr_bs  = np.concatenate(corr_bs_all)
    corr_inc = np.concatenate(corr_inc_all)

    curve_raw  = build_arc_curve(raw_bs,  raw_inc)
    curve_corr = build_arc_curve(corr_bs, corr_inc)

    # normalise raw to same mean as corrected for visual comparison
    valid_r = np.isfinite(curve_raw)
    valid_c = np.isfinite(curve_corr)
    curve_raw_norm = curve_raw - np.nanmean(curve_raw[valid_r]) + np.nanmean(curve_corr[valid_c])

    rms_before = arc_rms(curve_raw_norm)
    rms_after  = arc_rms(curve_corr)
    print(f"ARC RMS  before: {rms_before:.4f} dB")
    print(f"ARC RMS  after:  {rms_after:.4f} dB")
    print(f"Improvement:     {rms_before - rms_after:.4f} dB")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"ARC Verification — SSS {args.freq.upper()}", fontsize=13)

    # left: before vs after
    ax = axes[0]
    ax.plot(BIN_CENTERS, curve_raw_norm, label=f"Before  (RMS={rms_before:.3f} dB)", color="steelblue", linewidth=2)
    ax.plot(BIN_CENTERS, curve_corr,     label=f"After   (RMS={rms_after:.3f} dB)",  color="darkorange", linewidth=2)
    ax.axhline(np.nanmean(curve_corr[valid_c]), color="gray", linestyle="--", linewidth=1, label="Mean")
    ax.set_xlabel("Incidence angle (deg)")
    ax.set_ylabel("BS (dB)")
    ax.set_title("Global ARC: Before vs After")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # right: residual (how flat is the corrected curve)
    ax = axes[1]
    residual = curve_corr - np.nanmean(curve_corr[valid_c])
    ax.plot(BIN_CENTERS, residual, color="darkorange", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.fill_between(BIN_CENTERS, residual, 0,
                    where=np.isfinite(residual), alpha=0.2, color="darkorange")
    ax.set_xlabel("Incidence angle (deg)")
    ax.set_ylabel("Residual (dB)")
    ax.set_title(f"Corrected ARC residual  (RMS={rms_after:.3f} dB)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_fig = ROOT / f"outputs/figures/arc_verification_{args.freq}.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_fig}")
    plt.show()


if __name__ == "__main__":
    main()