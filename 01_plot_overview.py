# 01_plot_overview.py
import os
import glob
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colormaps
import survey_utils as utils  # 引用剛剛寫好的工具庫

# --- 設定區 (Config) ---
XYZ_PATH = r'data\multibeam\G1m_142m.txt'
JSF_DIR = r'data\sbp'
FILE_LIMIT = 10
FIG_SIZE = (12, 10)

def main():
    # 1. 準備底圖
    df = utils.read_xyz(XYZ_PATH)
    bg_lon, bg_lat = utils.twd97_to_wgs84(df['x'].values, df['y'].values)

    # 2. 準備測線
    jsf_files = sorted(glob.glob(os.path.join(JSF_DIR, "*.jsf")))[:FILE_LIMIT]
    
    # 3. 繪圖
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=120)
    
    # A. 畫水深
    sc = ax.scatter(bg_lon, bg_lat, c=df['z'], s=0.01, cmap='viridis')
    plt.colorbar(sc, label='Depth (m)')
    
    # B. 畫測線 (使用 tab20 色階)
    cmap = colormaps['tab20']
    
    print(f"Plotting {len(jsf_files)} tracks...")
    for i, jsf in enumerate(jsf_files):
        lons, lats, _ = utils.read_jsf_track(jsf)
        if len(lons) > 0:
            color = cmap(i % 20)
            label = f"{i+1}"
            
            # 畫線
            ax.plot(lons, lats, color=color, linewidth=2, label=label)
            # 畫起點圓點
            ax.plot(lons[0], lats[0], 'o', color=color, markeredgecolor='white')

    # 4. 修飾與圖例
    ax.set_title("Survey Overview")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()