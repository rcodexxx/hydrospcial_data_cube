"""
Anchor-based CC calibration.

Sets CC such that the survey median RL falls at the Hamilton (1970)
centerline of the dominant cohesive sediment class. Used as an
alternative to direct BSB-based calibration when ground truth is
unavailable or geometric constraints prevent stable BSB extraction.
"""
import numpy as np
from tqdm import tqdm

from src.config import get_config, ROOT
from src.sbp.config import BLANKING_SAMPLES
from src.sbp.read_sbp_jsf import read_sbp_jsf

# Hamilton (1970) class centerline RL for Clayey silt / Silty clay
ANCHOR_RL_DB = 14.6


def main():
    cfg = get_config()
    sbp_dirs = [ROOT / d["path"] for d in cfg["sbp"]["survey_dirs"]]
    jsf_files = []
    for d in sbp_dirs:
        jsf_files.extend(sorted(d.glob("*.jsf")))

    print(f"Anchor calibration: median RL → {ANCHOR_RL_DB} dB "
          f"(Clayey silt / Silty clay centerline, Hamilton 1970)\n")

    all_amp_b = []
    for jsf in tqdm(jsf_files, desc="Reading"):
        data = read_sbp_jsf(jsf)
        if "SBP" not in data:
            continue
        d = data["SBP"]
        valid = ~np.isnan(d["lon"])
        if valid.sum() == 0:
            continue
        amps = d["amps"][valid]
        search = amps[:, BLANKING_SAMPLES:]
        idx_b = np.argmax(search, axis=1) + BLANKING_SAMPLES
        amp_b = amps[np.arange(len(amps)), idx_b]
        all_amp_b.append(amp_b[amp_b > 0])

    all_amp_b = np.concatenate(all_amp_b)
    median_amp = float(np.median(all_amp_b))
    cc = median_amp * 10 ** (ANCHOR_RL_DB / 20)

    print(f"\nSurvey-wide amp_b median: {median_amp:.1f}")
    print(f"Anchor RL: {ANCHOR_RL_DB} dB")
    print(f"CC = median_amp × 10^(RL/20) = {cc:.4e}")
    print(f"\nUpdate configs/mudan.yaml sbp.calibration_constant:")
    print(f"  calibration_constant: {cc:.4e}")


if __name__ == "__main__":
    main()