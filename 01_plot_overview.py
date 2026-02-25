import os
import glob
import matplotlib.pyplot as plt
from matplotlib import colormaps
import utils  # 引用最新的 utils

# --- 設定區 (Config) ---
XYZ_PATH = r'data\multibeam\G1m_142m.txt'
JSF_DIR = r'data\sbp'
FILE_LIMIT = 51         # 只畫前 10 個檔案 (測試用)
FIG_SIZE = (12, 10)

def main():
    # 1. 準備底圖 (Multibeam)
    # 檢查檔案是否存在
    if os.path.exists(XYZ_PATH):
        print("Reading Multibeam background...")
        df = utils.read_xyz(XYZ_PATH)
        bg_lon, bg_lat = utils.twd97_to_wgs84(df['x'].values, df['y'].values)
    else:
        print("Multibeam file not found, skipping background.")
        df, bg_lon, bg_lat = None, [], []

    # 2. 準備 JSF 測線檔案
    jsf_files = sorted(glob.glob(os.path.join(JSF_DIR, "*.jsf")))[:FILE_LIMIT]
    
    # 3. 繪圖初始化
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=120)
    
    # A. 畫水深底圖
    if len(bg_lon) > 0:
        sc = ax.scatter(bg_lon, bg_lat, c=df['z'], s=1, cmap='viridis', alpha=0.5)
        plt.colorbar(sc, label='Depth (m)')
    
    # B. 畫測線
    # 取得 Matplotlib 的顏色表 (Tab20 適合區分不同測線)
    cmap = colormaps['tab20']
    
    print(f"Plotting {len(jsf_files)} tracks...")
    
    for i, jsf in enumerate(jsf_files):
        # --- 關鍵修改開始 ---
        # 使用新的 read_jsf 讀取資料
        data = utils.read_jsf(jsf)
        
        # 如果檔案是空的或沒讀到資料，就跳過
        if not data:
            continue
            
        # 從 List of Dicts 中提取經緯度
        lons = [p['lon'] for p in data]
        lats = [p['lat'] for p in data]
        # --- 關鍵修改結束 ---

        # 設定顏色與標籤
        color = cmap(i % 20)
        label = f"Line {i+1}"
        
        # 畫線
        ax.plot(lons, lats, color=color, linewidth=2, label=label)
        
        # 畫起點 (加個白邊讓它明顯一點)
        ax.plot(lons[0], lats[0], 'o', color=color, markeredgecolor='white', markersize=6)

    # 4. 圖表修飾
    ax.set_title("Survey Tracks Overview")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.axis('equal')  # 保持地圖比例正確
    ax.grid(True, linestyle='--', alpha=0.3)
    # ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1)) # 圖例放外面避免擋住地圖
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()