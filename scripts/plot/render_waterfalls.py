# scripts/render_waterfalls.py
"""
Batch render SSS waterfall and SBP envelope images.

SSS: port + stbd merged (nadir in center), one image per freq (HF, LF)
SBP: envelope profile (x=along-track, y=depth)

Output:
  outputs/waterfalls/sss/{filename}_HF.png
  outputs/waterfalls/sss/{filename}_LF.png
  outputs/waterfalls/sbp/{filename}.png
  outputs/waterfalls/index.json  ← metadata for viewer
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from src.sss.read_sss_jsf import read_sss_jsf
from src.sbp.read_sbp_jsf import read_sbp_jsf

ROOT = Path(__file__).parent.parent

SSS_DIRS = [
    ROOT / "data/sss/20251223",
    ROOT / "data/sss/20251224",
]
SBP_DIR = ROOT / "data/sbp"

OUT_SSS = ROOT / "outputs/waterfalls/sss"
OUT_SBP = ROOT / "outputs/waterfalls/sbp"
OUT_INDEX = ROOT / "outputs/waterfalls/index.json"

OUT_SSS.mkdir(parents=True, exist_ok=True)
OUT_SBP.mkdir(parents=True, exist_ok=True)

DPI = 100
MAX_WIDTH_PX = 1400


def render_sss(jsf_path):
    """Render merged SSS waterfall for HF and LF."""
    try:
        data = read_sss_jsf(jsf_path)
    except Exception as e:
        print(f"  Error reading {jsf_path.name}: {e}")
        return []

    results = []
    stem = jsf_path.stem

    for freq in ["HF", "LF"]:
        port_ch = f"{freq}_port"
        stbd_ch = f"{freq}_stbd"
        if port_ch not in data or stbd_ch not in data:
            continue

        port = data[port_ch]["amps"]
        stbd = data[stbd_ch]["amps"]

        n_pings = min(port.shape[0], stbd.shape[0])
        if n_pings < 5:
            continue

        port = port[:n_pings]
        stbd = stbd[:n_pings]

        # merge: port flipped | stbd
        merged = np.hstack([port[:, ::-1], stbd])

        valid = merged[merged > 0]
        if len(valid) < 100:
            continue
        vmin, vmax = np.percentile(valid, [2, 98])

        # figure sizing: width fixed, height proportional to pings
        n_samples = merged.shape[1]
        fig_w = min(MAX_WIDTH_PX / DPI, 14)
        fig_h = max(4, n_pings / 40)
        fig_h = min(fig_h, 80)  # cap height

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(merged, aspect="auto", cmap="copper",
                  vmin=vmin, vmax=vmax, interpolation="none")

        nadir_x = port.shape[1]
        ax.axvline(nadir_x, color="cyan", linewidth=0.5, alpha=0.4)

        ax.set_xlabel("← Port | Stbd →", fontsize=9)
        ax.set_ylabel("Ping", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title(f"{freq} — {stem}", fontsize=10)
        plt.tight_layout()

        out_path = OUT_SSS / f"{stem}_{freq}.png"
        plt.savefig(out_path, dpi=DPI)
        plt.close(fig)

        results.append({
            "file": jsf_path.name,
            "freq": freq,
            "image": f"sss/{stem}_{freq}.png",
            "pings": n_pings,
            "samples_port": int(port.shape[1]),
            "samples_stbd": int(stbd.shape[1]),
        })

    return results


def render_sbp(jsf_path):
    """Render SBP envelope profile."""
    try:
        data = read_sbp_jsf(jsf_path)
    except Exception as e:
        print(f"  Error reading {jsf_path.name}: {e}")
        return None

    if "SBP" not in data:
        return None

    amps = data["SBP"]["amps"]
    n_pings, n_samples = amps.shape

    if n_pings < 5:
        return None

    valid = amps[amps > 0]
    if len(valid) < 100:
        return None
    vmin, vmax = np.percentile(valid, [2, 95])

    # x = along-track (pings), y = depth (samples, top=shallow)
    fig_w = max(8, n_pings / 30)
    fig_w = min(fig_w, 60)
    fig_h = 5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(amps.T, aspect="auto", cmap="gray_r",
              vmin=vmin, vmax=vmax, interpolation="none",
              origin="upper")
    ax.set_xlabel("Ping (along-track)", fontsize=9)
    ax.set_ylabel("Sample (depth ↓)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(f"SBP — {jsf_path.stem}", fontsize=10)
    plt.tight_layout()

    stem = jsf_path.stem
    out_path = OUT_SBP / f"{stem}.png"
    plt.savefig(out_path, dpi=DPI)
    plt.close(fig)

    return {
        "file": jsf_path.name,
        "image": f"sbp/{stem}.png",
        "pings": n_pings,
        "samples": n_samples,
    }


def main():
    index = {"sss": {}, "sbp": {}}

    # SSS
    sss_files = []
    for d in SSS_DIRS:
        if d.exists():
            sss_files.extend(sorted(d.glob("*.jsf")))

    print(f"Rendering SSS waterfalls ({len(sss_files)} files)...")
    for jsf in tqdm(sss_files, desc="SSS"):
        results = render_sss(jsf)
        for r in results:
            key = f"{jsf.name}_{r['freq']}"
            index["sss"][key] = r

    # SBP
    sbp_files = sorted(SBP_DIR.glob("*.jsf")) if SBP_DIR.exists() else []

    print(f"\nRendering SBP profiles ({len(sbp_files)} files)...")
    for jsf in tqdm(sbp_files, desc="SBP"):
        result = render_sbp(jsf)
        if result:
            index["sbp"][jsf.name] = result

    # save index
    with open(OUT_INDEX, "w") as f:
        json.dump(index, f, indent=2)

    n_sss = len(index["sss"])
    n_sbp = len(index["sbp"])
    print(f"\nDone:")
    print(f"  SSS images: {n_sss}")
    print(f"  SBP images: {n_sbp}")
    print(f"  Index: {OUT_INDEX}")


if __name__ == "__main__":
    main()