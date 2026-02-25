import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import utils
import os
import glob
import pandas as pd

# ================= 參數設定 =================
JSF_DIR = r'data\sbp'
MBES_FILE = r'data\multibeam\G1m_142m.txt'
CHANNEL = 0 

CC_VALUE = 8.5867e07

# --- 地質分類標準 (8類) ---
BOUNDARIES = [0, 8.2, 8.8, 9.5, 11.0, 12.8, 14.3, 15.6, 50]
COLORS = [
    '#8B0000', # Coarse Sand (<8.2)
    '#FF0000', # Fine Sand (8.2-8.8)
    '#FF8C00', # Very Fine Sand (8.8-9.5)
    '#FFD700', # Silty Sand (9.5-11.0)
    '#ADFF2F', # Sand-Silt-Clay (11.0-12.8)
    '#32CD32', # Sandy Silt (12.8-14.3)
    '#00CED1', # Clayey Silt (14.3-15.6)
    '#00008B'  # Silty Clay (>15.6)
]
LABELS = [
    "Coarse Sand (< 8.2 dB)",
    "Fine Sand (8.2-8.8 dB)",
    "Very Fine Sand (8.8-9.5 dB)",
    "Silty Sand (9.5-11.0 dB)",
    "Sand-Silt-Clay (11.0-12.8 dB)",
    "Sandy Silt (12.8-14.3 dB)",
    "Clayey Silt (14.3-15.6 dB)",
    "Silty Clay (> 15.6 dB)"
]

FIG_SIZE = (12, 9)
DPI = 100
SCATTER_SIZE = 2
# ===========================================

def get_sbp_data_final_formula(jsf_path):
    """
    RL = -20 * log10(R)
    """
    lons, lats, pings = utils.read_jsf_track(jsf_path)
    
    # 修復 NumPy Array 判空錯誤
    if len(lons) == 0: return [], [], []

    # 直接使用原始 GPS (No Offset)
    c_lons, c_lats = lons, lats
    
    file_rls = []
    file_lons = []
    file_lats = []
    
    # 全解析度讀取
    traces = utils.read_jsf_segment_data(jsf_path, pings[0], pings[-1], channel=CHANNEL)
    
    # 建立 ping map 加速
    ping_map = {p: i for i, p in enumerate(pings)}

    for packet in traces:
        amps = packet['amps']
        if len(amps) < 100: continue
        
        search_region = amps[50:]
        if len(search_region) == 0: continue
        
        idx_max = np.argmax(search_region) + 50
        amp_1 = float(amps[idx_max])
        r_1 = float(idx_max)

        if amp_1 > 0 and r_1 > 0:
            # R = (r * A) / CC
            # R = (r_1 * amp_1)
            R = (r_1 * amp_1) / CC_VALUE
            
            # 物理限制
            if R > 1.0: R = 1.0
            if R < 0.0001: R = 0.0001 # 避免 log(0)
            
            # 轉換為 RL (dB)
            # RL = -20 * np.log10(R) + 20 * np.log10(CC_VALUE)
            RL = -20 * np.log10(R)
            
            # 對應 GPS
            p_num = packet['ping_num']
            if p_num in ping_map:
                idx = ping_map[p_num]
                file_rls.append(RL)
                file_lons.append(c_lons[idx])
                file_lats.append(c_lats[idx])
            
    return file_lons, file_lats, file_rls

def main():
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    # 1. 繪製多音束地形底圖
    if os.path.exists(MBES_FILE):
        print(f"Loading Bathymetry: {os.path.basename(MBES_FILE)}...")
        try:
            df = pd.read_csv(MBES_FILE, sep=r'\s+', header=None, names=['x', 'y', 'z'])
            bg_lons, bg_lats = utils.twd97_to_wgs84(df['x'].values, df['y'].values)
            
            sc_bg = ax.scatter(bg_lons, bg_lats, c=df['z'], s=2, cmap='gray', alpha=0.4, label='Bathymetry')
            cbar = plt.colorbar(sc_bg, ax=ax, fraction=0.02, pad=0.04)
            cbar.set_label('Water Depth (m)')
        except Exception as e:
            print(f"Background Error: {e}")

    # 2. 繪製 SBP 疊圖
    jsf_files = sorted(glob.glob(os.path.join(JSF_DIR, "*.jsf")))
    
    all_lons, all_lats, all_rls = [], [], []
    for i, jsf in enumerate(jsf_files):
        l, lt, r = get_sbp_data_final_formula(jsf)
        all_lons.extend(l)
        all_lats.extend(lt)
        all_rls.extend(r)
    
    print(f"\nTotal Points: {len(all_rls)}")

    # 繪圖
    cmap_sbp = mcolors.ListedColormap(COLORS)
    norm_sbp = mcolors.BoundaryNorm(BOUNDARIES, cmap_sbp.N)
    
    sc_sbp = ax.scatter(all_lons, all_lats, c=all_rls, s=SCATTER_SIZE, 
                        cmap=cmap_sbp, norm=norm_sbp, alpha=1.0, zorder=3, edgecolors='none')

    # 圖例
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(COLORS, LABELS)]
    legend = ax.legend(handles=patches, title="Sediment Type")
    ax.add_artist(legend)
    
    ax.set_title(f"Sediment Classification Overlay")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Auto Zoom
    if all_lons:
        margin = 0.0005
        ax.set_xlim(min(all_lons)-margin, max(all_lons)+margin)
        ax.set_ylim(min(all_lats)-margin, max(all_lats)+margin)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()