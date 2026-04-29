"""
Validate anchor-based CC calibration against BSB-based CC and check
the dominant-sediment unimodality assumption.

Two checks:
  (1) Unimodality of log(amp_b) — does the survey-wide bottom amplitude
      distribution show a single dominant mode? If not, median anchoring
      is built on a false premise.
  (2) BSB cross-check — run estimate_cc on per-segment ping averages,
      compare CC_BSB distribution against CC_anchor. Two views:
        - all segments
        - "stable" segments only (lowest amp_b rolling std)

Outputs:
  fig_unimodality.png
  fig_bsb_validation.png
  Console summary with quantitative metrics.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from tqdm import tqdm

from src.config import get_config, ROOT
from src.sbp.config import BLANKING_SAMPLES
from src.sbp.read_sbp_jsf import read_sbp_jsf
from src.sbp.calculation import estimate_cc

ANCHOR_RL_DB = 14.6           # Clayey silt / Silty clay centerline (freshwater)
SEGMENT_SIZE = 100            # ping per segment for BSB averaging
STABLE_FRACTION = 0.2         # take lowest 20% std segments as "stable"
ROLLING_WINDOW = 50           # for amp_b std along track


def collect_pings():
    """Read all JSF, return concatenated amp_b and amps_2d (per file list)."""
    cfg = get_config()
    sbp_dirs = [ROOT / d["path"] for d in cfg["sbp"]["survey_dirs"]]
    jsf_files = []
    for d in sbp_dirs:
        jsf_files.extend(sorted(d.glob("*.jsf")))

    all_amp_b = []
    file_segments = []   # list of (amps_2d, file_name)

    for jsf in tqdm(jsf_files, desc="Reading"):
        data = read_sbp_jsf(jsf)
        if "SBP" not in data:
            continue
        d = data["SBP"]
        valid = ~np.isnan(d["lon"])
        if valid.sum() < SEGMENT_SIZE:
            continue
        amps = d["amps"][valid]
        search = amps[:, BLANKING_SAMPLES:]
        idx_b = np.argmax(search, axis=1) + BLANKING_SAMPLES
        amp_b = amps[np.arange(len(amps)), idx_b]
        ok = amp_b > 0
        if ok.sum() < SEGMENT_SIZE:
            continue
        all_amp_b.append(amp_b[ok])
        file_segments.append((amps[ok], jsf.name))

    return np.concatenate(all_amp_b), file_segments


# ──────────────────────────────────────────────────────────
# (1) Unimodality check
# ──────────────────────────────────────────────────────────
def check_unimodality(amp_b):
    log_amp = np.log10(amp_b)
    median_log = np.median(log_amp)

    # KDE for mode detection
    kde = gaussian_kde(log_amp)
    x = np.linspace(log_amp.min(), log_amp.max(), 1000)
    y = kde(x)
    mode_log = x[np.argmax(y)]

    # Count significant peaks (> 30% of max density)
    peaks, _ = find_peaks(y, height=0.3 * y.max())
    n_modes = len(peaks)

    median_mode_diff_db = 20.0 * (median_log - mode_log)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(log_amp, bins=80, density=True, alpha=0.4, color="steelblue",
            label="histogram")
    ax.plot(x, y, "k-", lw=1.5, label="KDE")
    ax.axvline(median_log, color="red", ls="--", lw=1.5,
               label=f"median = {median_log:.3f}")
    ax.axvline(mode_log, color="green", ls="--", lw=1.5,
               label=f"mode = {mode_log:.3f}")
    for p in peaks:
        ax.plot(x[p], y[p], "v", color="orange", ms=10)
    ax.set_xlabel("log10(amp_b)")
    ax.set_ylabel("density")
    ax.set_title(
        f"Unimodality check: {n_modes} mode(s) detected, "
        f"median−mode = {median_mode_diff_db:.2f} dB"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_unimodality.png", dpi=120)
    plt.close(fig)

    return {
        "n_modes": n_modes,
        "median_log": median_log,
        "mode_log": mode_log,
        "median_mode_diff_db": median_mode_diff_db,
        "n_pings": len(amp_b),
    }


# ──────────────────────────────────────────────────────────
# (2) BSB cross-check
# ──────────────────────────────────────────────────────────
def estimate_cc_per_segment(file_segments):
    """Run estimate_cc on N-ping segments. Also compute amp_b std per
    segment as a stability metric."""
    cc_list = []
    std_list = []
    success_flags = []

    for amps, _ in file_segments:
        n_seg = len(amps) // SEGMENT_SIZE
        for i in range(n_seg):
            seg = amps[i * SEGMENT_SIZE:(i + 1) * SEGMENT_SIZE]
            cc = estimate_cc(seg)

            # amp_b std as stability metric
            search = seg[:, BLANKING_SAMPLES:]
            idx_b = np.argmax(search, axis=1) + BLANKING_SAMPLES
            amp_b = seg[np.arange(len(seg)), idx_b]
            std_list.append(np.std(np.log10(amp_b[amp_b > 0] + 1e-12)))

            if cc is not None and cc > 0:
                cc_list.append(cc)
                success_flags.append(True)
            else:
                cc_list.append(np.nan)
                success_flags.append(False)

    return np.array(cc_list), np.array(std_list), np.array(success_flags)


def plot_bsb_validation(cc_bsb, stds, ok, cc_anchor):
    cc_bsb_db = 20 * np.log10(cc_bsb[ok])
    cc_anchor_db = 20 * np.log10(cc_anchor)

    # Stable subset: lowest std segments where BSB also succeeded
    stable_thresh = np.quantile(stds[ok], STABLE_FRACTION)
    stable_mask = ok & (stds <= stable_thresh)
    cc_stable_db = 20 * np.log10(cc_bsb[stable_mask])

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))

    # Top: distribution
    ax = axes[0]
    ax.hist(cc_bsb_db, bins=40, alpha=0.5, color="steelblue",
            label=f"all segments (n={ok.sum()})")
    ax.hist(cc_stable_db, bins=40, alpha=0.6, color="orange",
            label=f"stable segments (n={stable_mask.sum()})")
    ax.axvline(cc_anchor_db, color="red", ls="--", lw=2,
               label=f"CC_anchor = {cc_anchor_db:.2f} dB")
    ax.axvline(np.median(cc_bsb_db), color="steelblue", ls=":", lw=1.5,
               label=f"median(all) = {np.median(cc_bsb_db):.2f} dB")
    if stable_mask.sum() > 5:
        ax.axvline(np.median(cc_stable_db), color="orange", ls=":", lw=1.5,
                   label=f"median(stable) = {np.median(cc_stable_db):.2f} dB")
    ax.set_xlabel("CC (dB)")
    ax.set_ylabel("count")
    ax.set_title("BSB cross-check: CC distribution vs anchor")
    ax.legend()

    # Bottom: scatter along segments
    ax = axes[1]
    seg_idx = np.arange(len(cc_bsb))
    ax.scatter(seg_idx[ok & ~stable_mask], 20 * np.log10(cc_bsb[ok & ~stable_mask]),
               s=8, alpha=0.4, color="steelblue", label="all (BSB ok)")
    ax.scatter(seg_idx[stable_mask], 20 * np.log10(cc_bsb[stable_mask]),
               s=12, alpha=0.7, color="orange", label="stable")
    ax.axhline(cc_anchor_db, color="red", ls="--", lw=2, label="CC_anchor")
    ax.set_xlabel("segment index")
    ax.set_ylabel("CC_BSB (dB)")
    ax.set_title("CC_BSB per segment")
    ax.legend()

    fig.tight_layout()
    fig.savefig("fig_bsb_validation.png", dpi=120)
    plt.close(fig)

    return {
        "n_segments": len(cc_bsb),
        "n_bsb_success": int(ok.sum()),
        "success_rate": float(ok.mean()),
        "cc_anchor_db": float(cc_anchor_db),
        "cc_bsb_median_db_all": float(np.median(cc_bsb_db)),
        "cc_bsb_std_db_all": float(np.std(cc_bsb_db)),
        "cc_bsb_median_db_stable": (
            float(np.median(cc_stable_db)) if stable_mask.sum() > 5 else None
        ),
        "cc_bsb_std_db_stable": (
            float(np.std(cc_stable_db)) if stable_mask.sum() > 5 else None
        ),
    }


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    print("Reading SBP data...")
    amp_b_all, file_segments = collect_pings()
    print(f"Total pings: {len(amp_b_all)}\n")

    # CC_anchor from median amp_b
    median_amp = float(np.median(amp_b_all))
    cc_anchor = median_amp * 10 ** (ANCHOR_RL_DB / 20)

    # ── (1) Unimodality ──
    print("=" * 60)
    print("(1) Unimodality check")
    print("=" * 60)
    uni = check_unimodality(amp_b_all)
    print(f"  pings analysed       : {uni['n_pings']}")
    print(f"  modes detected       : {uni['n_modes']}")
    print(f"  median (log10 amp_b) : {uni['median_log']:.4f}")
    print(f"  mode   (log10 amp_b) : {uni['mode_log']:.4f}")
    print(f"  median - mode        : {uni['median_mode_diff_db']:.2f} dB")
    print()
    if uni['n_modes'] == 1:
        print("  -> unimodal. Dominant-sediment assumption supported.")
    else:
        print("  -> WARNING: multimodal. Median anchoring premise may fail.")
    if abs(uni['median_mode_diff_db']) > 1.0:
        print("  -> NOTE: median deviates from mode by >1 dB. Consider "
              "anchoring to mode instead.")
    print(f"  Saved: fig_unimodality.png\n")

    # ── (2) BSB cross-check ──
    print("=" * 60)
    print("(2) BSB cross-check")
    print("=" * 60)
    cc_bsb, stds, ok = estimate_cc_per_segment(file_segments)
    res = plot_bsb_validation(cc_bsb, stds, ok, cc_anchor)
    print(f"  segments analysed    : {res['n_segments']}")
    print(f"  BSB extraction OK    : {res['n_bsb_success']} "
          f"({100 * res['success_rate']:.1f}%)")
    print(f"  CC_anchor            : {res['cc_anchor_db']:.2f} dB")
    print(f"  CC_BSB median (all)  : {res['cc_bsb_median_db_all']:.2f} dB "
          f"(std {res['cc_bsb_std_db_all']:.2f} dB)")
    if res['cc_bsb_median_db_stable'] is not None:
        print(f"  CC_BSB median (stable): {res['cc_bsb_median_db_stable']:.2f} dB "
              f"(std {res['cc_bsb_std_db_stable']:.2f} dB)")

    diff_all = res['cc_anchor_db'] - res['cc_bsb_median_db_all']
    print(f"\n  anchor − BSB(all)    : {diff_all:+.2f} dB")
    if res['cc_bsb_median_db_stable'] is not None:
        diff_stable = res['cc_anchor_db'] - res['cc_bsb_median_db_stable']
        print(f"  anchor − BSB(stable) : {diff_stable:+.2f} dB")

    print()
    if abs(diff_all) < 2.0:
        print("  -> anchor and BSB(all) agree within 2 dB. Strong "
              "independent support.")
    elif abs(diff_all) < 5.0:
        print("  -> moderate disagreement (2-5 dB). Methods broadly consistent "
              "but caveat in paper.")
    else:
        print("  -> WARNING: large disagreement (>5 dB). One method is biased; "
              "investigate further.")
    print(f"  Saved: fig_bsb_validation.png")


if __name__ == "__main__":
    main()