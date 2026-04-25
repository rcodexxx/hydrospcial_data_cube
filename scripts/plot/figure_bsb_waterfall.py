"""
Figure: BSB geometry on waterfall.

Method figure (2D waterfall view) complementing figure_bsb_geometry.py.
Shows per-ping envelope evolution with B and BSB overlaid; the BSB
line follows the per-ping argmax within the search window.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from src.config import get_config, ROOT
from src.sbp.read_sbp_jsf import read_sbp_jsf
from src.sbp.config import BLANKING_SAMPLES

STABLE_JSF = "20251223034109.jsf"
PING_START = 550
PING_END = 650


def main():
    cfg = get_config()
    sbp_dir = ROOT / cfg["sbp"]["survey_dirs"][0]["path"]
    jsf_path = sbp_dir / STABLE_JSF

    data = read_sbp_jsf(jsf_path)["SBP"]
    amps = data["amps"][PING_START:PING_END].astype(np.float64)
    n_pings, n_samples = amps.shape

    envelopes = np.abs(hilbert(amps, axis=1))
    db = 20 * np.log10(np.maximum(envelopes, 1.0))
    db_norm = db - db.max(axis=1, keepdims=True)

    # Per-ping B and BSB
    idx_b_arr = np.argmax(envelopes[:, BLANKING_SAMPLES:], axis=1) + BLANKING_SAMPLES
    idx_bsb_arr = np.zeros(n_pings, dtype=int)
    for i in range(n_pings):
        lo = int(idx_b_arr[i] * 1.95)
        hi = int(idx_b_arr[i] * 2.10)
        if hi < n_samples:
            idx_bsb_arr[i] = lo + int(np.argmax(envelopes[i, lo:hi]))
        else:
            idx_bsb_arr[i] = lo

    y_lo = max(0, int(idx_b_arr.min()) - 80)
    y_hi = min(n_samples, int(idx_bsb_arr.max()) + 150)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(db_norm[:, y_lo:y_hi].T,
                   aspect='auto', cmap='gray_r',
                   extent=[0, n_pings, y_hi, y_lo],
                   vmin=-30, vmax=-5,         # Tighter range to enhance layer visibility
                   interpolation='nearest')

    pings_x = np.arange(n_pings)
    ax.plot(pings_x, idx_b_arr, color='crimson',
            linewidth=2, alpha=0.9, label='B')
    ax.plot(pings_x, idx_bsb_arr, color='royalblue',
            linewidth=2, alpha=0.9, label='BSB')

    ax.set_xlabel('Ping number (in segment)')
    ax.set_ylabel('Sample index')
    ax.set_title(
        f'Envelope waterfall — {STABLE_JSF[8:14]} pings {PING_START}-{PING_END}'
    )
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.95)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Envelope amplitude (dB, per-ping ref.)', fontsize=9)

    out_path = ROOT / "outputs/figures/figure_bsb_waterfall.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    print(f"  B median  : {int(np.median(idx_b_arr))}")
    print(f"  BSB median: {int(np.median(idx_bsb_arr))}")


if __name__ == "__main__":
    main()