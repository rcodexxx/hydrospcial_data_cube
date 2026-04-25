# scripts/build/extract_sbp_points.py
"""
Extract pure, un-interpolated SBP trackline data (Ground Truth).

Workflow:
  1. Read raw JSF files for SBP data.
  2. Extract lon/lat and convert to target EPSG (X, Y).
  3. Apply CC calibration to calculate absolute Reflection Loss (RL in dB).
  4. Lookup Hamilton table to get sediment classification.
  5. Save as a pure point-cloud CSV for cross-sensor anchoring and RF training.

Outputs:
  sbp_tracklines.csv - Contains X, Y, RL_dB, Sediment_ID, Sediment_Name
"""
import pandas as pd
import numpy as np
from pathlib import Path
from pyproj import Transformer
from tqdm import tqdm

# 匯入你原本專案架構內的模組
from src.sbp.read_sbp_jsf import read_sbp_jsf
from src.sbp.calculation import SEDIMENT_LABELS, classify_sediment, compute_rl
from src.config import SBP_CC, EPSG, ROOT

# ==========================================
# 參數設定
# ==========================================
SBP_PATH = ROOT / "data/sbp"
OUT_CSV = ROOT / "outputs/csv/sbp_tracklines.csv"

def main():
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)
    cc = SBP_CC
    print(f"Using calibrated CC = {cc:.6e} ({20 * np.log10(cc):.2f} dB)")

    # 1. 萃取所有測線的絕對物理值 (RL)
    all_x, all_y, all_rl = [], [], []

    jsf_files = sorted(SBP_PATH.glob("*.jsf"))
    if not jsf_files:
        print(f"Error: No JSF files found in {SBP_PATH}")
        return

    for jsf in tqdm(jsf_files, desc="Extracting SBP Tracklines"):
        data = read_sbp_jsf(jsf)
        if "SBP" not in data:
            continue
            
        d = data["SBP"]
        valid = ~np.isnan(d["lon"])
        if valid.sum() == 0:
            continue
            
        # 轉換座標
        x, y = transformer.transform(d["lon"][valid], d["lat"][valid])
        amps = d["amps"][valid]

        # 逐點計算 RL
        for i in range(len(amps)):
            rl = compute_rl(amps[i], cc)
            if rl is None:
                continue
            all_x.append(x[i])
            all_y.append(y[i])
            all_rl.append(rl)

    if not all_rl:
        print("No valid SBP data could be extracted.")
        return

    print(f"\nExtracted {len(all_rl):,} valid ground-truth pings.")
    
    # 2. 進行 Hamilton 底質分類
    print("Classifying sediments via Hamilton model...")
    sed_ids = [classify_sediment(rl) for rl in all_rl]
    sed_names = [SEDIMENT_LABELS[sid] if 0 <= sid < len(SEDIMENT_LABELS) else "Unknown" for sid in sed_ids]

    # 3. 儲存成乾淨的點位 CSV 檔案
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame({
        'X': all_x,
        'Y': all_y,
        'RL_dB': all_rl,
        'Sediment_ID': sed_ids,
        'Sediment_Name': sed_names
    })
    
    df.to_csv(OUT_CSV, index=False)
    
    print(f"\n✅ Successfully saved pure tracklines to: {OUT_CSV}")
    print(f"   RL Range: {min(all_rl):.2f} ~ {max(all_rl):.2f} dB")
    print(f"   Top 3 Sediments:")
    print(df['Sediment_Name'].value_counts().head(3).to_string())

if __name__ == "__main__":
    main()