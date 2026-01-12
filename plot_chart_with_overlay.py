import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm # 引入 colormap 模組
from pyproj import Transformer
import struct
import os
import glob

# ================= 參數設定 =================
XYZ_FILE = r'data\multibeam\G1m_142m.txt'
JSF_FOLDER = r'data\sbp'
FILE_LIMIT = 10

# 畫布設定 (稍微加寬一點，留空間給右邊的圖例)
FIG_SIZE = (12, 10) 
DPI = 100

# 顏色設定
BG_CMAP = 'gray'        # 底圖水深顏色
TRACK_WIDTH = 2.0          # 測線加粗一點比較好看清楚顏色
# 使用 'tab20' 色階，專門用於區分不同類別 (最多20色)
TRACK_COLORMAP = 'tab20'   

CRS_XYZ = "epsg:3826"
CRS_GPS = "epsg:4326"
# ===========================================

def read_jsf_nav(jsf_path):
    # (此函式內容與之前相同，省略以節省篇幅)
    lons, lats = [], []
    try:
        with open(jsf_path, 'rb') as f:
            while True:
                header = f.read(16)
                if not header: break
                try:
                    _, _, _, msg_type = struct.unpack_from('<HBBH', header)
                    size = struct.unpack_from('<I', header, 12)[0]
                except: break
                if msg_type == 80:
                    type80 = f.read(240)
                    rx = struct.unpack_from('<i', type80, 80)[0]
                    ry = struct.unpack_from('<i', type80, 84)[0]
                    units = struct.unpack_from('<h', type80, 88)[0]
                    if rx != 0 and ry != 0:
                        if units == 2:
                            lons.append(rx / 600000.0)
                            lats.append(ry / 600000.0)
                        elif units == 1:
                            lons.append(rx)
                            lats.append(ry)
                    f.seek(size - 240, 1)
                else:
                    f.seek(size, 1)
    except: pass
    return lons, lats

def plot_survey_colored_legend():
    try:
        print("1. 準備底圖...")
        df = pd.read_csv(XYZ_FILE, sep=r'\s+', header=None, names=['x', 'y', 'z'])
        trans = Transformer.from_crs(CRS_XYZ, CRS_GPS, always_xy=True)
        lon, lat = trans.transform(df['x'].values, df['y'].values)

        # 建立畫布
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

        print("2. 繪製水深底圖...")
        sc = ax.scatter(lon, lat, c=df['z'], s=0.01, cmap=BG_CMAP, zorder=1)
        cbar = plt.colorbar(sc, ax=ax, label='Depth (m)', pad=0.02)

        # 準備測線資料
        jsf_files = sorted(glob.glob(os.path.join(JSF_FOLDER, "*.jsf")))[:FILE_LIMIT]
        num_files = len(jsf_files)
        print(f"3. 開始繪製 {num_files} 條彩色測線...")

        # 取得分類專用色階物件
        cmap = matplotlib.colormaps[TRACK_COLORMAP]

        # 迴圈繪製每一條線
        for i, jsf in enumerate(jsf_files):
            jlons, jlats = read_jsf_nav(jsf)
            
            if jlons:
                file_name = os.path.basename(jsf)
                
                # --- 關鍵核心 ---
                # 從色階中取出第 i 個顏色
                # i % 20 是為了防止檔案超過 20 個時顏色索引超出範圍，讓它循環使用
                this_color = cmap(i % 20)
                
                # 繪圖時加入 'label' 參數，這是圖例顯示的內容
                # 我們在檔名前面加個數字編號，對照更方便
                label_str = f"{i+1}"

                # 畫線 (設定顏色和標籤)
                # zorder=2 確保線條疊在底圖上面
                ax.plot(jlons, jlats, color=this_color, linewidth=TRACK_WIDTH, 
                        label=label_str, zorder=2)
                
                # 加強起點顯示 (用同樣的顏色畫一個帶白邊的圓點)
                ax.plot(jlons[0], jlats[0], marker='o', color=this_color, 
                        markeredgecolor='white', markersize=6, zorder=3)

        # --- 建立圖例 (Legend) ---
        print("4. 產生圖例...")
        ax.legend(title="Trackline Files", borderaxespad=0., frameon=True)

        ax.set_title(f"Survey Map with {num_files} Colored Tracklines")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.axis('equal')
        ax.grid(True, linestyle='--', alpha=0.3)

        # 自動調整版面，避免圖例被切掉
        plt.tight_layout()
        plt.show()

    except Exception as e:
        # print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 需要匯入 matplotlib 才能使用 colormaps
    import matplotlib 
    plot_survey_colored_legend()