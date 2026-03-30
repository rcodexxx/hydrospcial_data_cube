# scripts/plot_sbp_profile.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_loader.read_sbp_jsf import read_sbp_jsf

SBP_PATH = Path("../../data/sbp")
TARGET_FILE = "20251223022116.jsf"
TARGET_START = 3770
TARGET_END = 3901

NS_PER_SAMPLE = 20480e-9
SOUND_SPEED = 1500.0
SAMPLE_DEPTH = NS_PER_SAMPLE * SOUND_SPEED / 2
MAX_DEPTH = 80.0


def find_reflections(amps):
    blanking = 50

    # B
    idx_b = int(np.argmax(amps[blanking:])) + blanking

    # BS: just after B, suppress sidelobes
    bs_start = idx_b + 30
    bs_end = min(len(amps), idx_b + 600)
    window = amps[bs_start:bs_end].copy()
    window[: min(80, len(window))] = 0
    idx_bs = int(np.argmax(window)) + bs_start

    # BSB: at ~2x B
    center = idx_b * 2
    bsb_start = max(int(idx_b * 1.85), center - 150)
    bsb_end = min(len(amps), center + 150)
    idx_bsb = int(np.argmax(amps[bsb_start:bsb_end])) + bsb_start

    return idx_b, idx_bs, idx_bsb


def main():
    data = read_sbp_jsf(SBP_PATH / TARGET_FILE)["SBP"]
    valid = ~np.isnan(data["lon"])
    orig_idx = np.where(valid)[0]
    sel = orig_idx[(orig_idx >= TARGET_START) & (orig_idx < TARGET_END)]

    amps_matrix = data["amps"][sel].astype(np.float32)
    n_pings = len(sel)
    max_sample = int(MAX_DEPTH / SAMPLE_DEPTH)
    amps_clip = amps_matrix[:, :max_sample]
    depth_axis = np.arange(max_sample) * SAMPLE_DEPTH

    # Reflections per ping
    d_B = np.full(n_pings, np.nan)
    d_BS = np.full(n_pings, np.nan)
    d_BSB = np.full(n_pings, np.nan)

    for i in range(n_pings):
        idx_b, idx_bs, idx_bsb = find_reflections(amps_matrix[i])
        d_B[i] = idx_b * SAMPLE_DEPTH
        d_BS[i] = idx_bs * SAMPLE_DEPTH
        d_BSB[i] = idx_bsb * SAMPLE_DEPTH

    # Clip to display range
    for arr in [d_B, d_BS, d_BSB]:
        arr[arr > MAX_DEPTH] = np.nan

    # Convert to dB
    amps_db = 20 * np.log10(np.maximum(amps_clip, 1.0))
    valid_db = amps_db[amps_db > 20]
    vmin = float(np.percentile(valid_db, 5))
    vmax = float(np.percentile(valid_db, 98))

    fig, ax = plt.subplots(figsize=(14, 6), layout="tight")

    im = ax.imshow(
        amps_db.T,
        aspect="auto",
        cmap="Greys",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        extent=[0, n_pings, depth_axis[-1], depth_axis[0]],
    )
    ax.plot(np.arange(n_pings), d_B, "r-", linewidth=3, label="B")
    ax.plot(np.arange(n_pings), d_BS, "g--", linewidth=3, label="BS")
    ax.plot(np.arange(n_pings), d_BSB, "o:", linewidth=3, label="BSB")

    ax.set_xlabel("Ping number", fontsize=11)
    ax.set_ylabel("Sonar range (m)", fontsize=11)
    ax.set_ylim(MAX_DEPTH, 0)
    ax.set_xlim(0, n_pings)
    ax.legend(loc="lower right", fontsize=10)
    plt.colorbar(im, ax=ax, label="Amplitude (dB)", pad=0.01)

    plt.savefig("../outputs/figures/sbp_profile_fig7.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
