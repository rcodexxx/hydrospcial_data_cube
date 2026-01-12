import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
import struct
import os
import glob

# ================= 參數設定 =================
XYZ_FILE = r'data\multibeam\G1m_142m.txt'
JSF_FOLDER = r'data\sbp'
TARGET_FILE_INDEX = 5   # 想看第幾個檔案 (0 = 第一個)

# 刻度設定
MARKER_INTERVAL = 100   # 每隔多少公尺標示一個數字 (例如 100m)
MARKER_COLOR = 'white'  # 字體顏色
MARKER_BG = 'red'       # 字體背景色

# 繪圖設定
FIG_SIZE = (12, 10)     # 圖畫大一點比較好看清楚數字
DPI = 150
XYZ_CMAP = 'viridis'

CRS_XYZ = "epsg:3826"
CRS_GPS = "epsg:4326"
# ===========================================

def read_jsf_data(jsf_path):
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
    return np.array(lons), np.array(lats)

def calculate_cumulative_distance(lons, lats):
    """ 計算累積距離陣列 (回傳單位: 公尺) """
    transformer = Transformer.from_crs(CRS_GPS, CRS_XYZ, always_xy=True)
    xx, yy = transformer.transform(lons, lats)
    
    dx = np.diff(xx)
    dy = np.diff(yy)
    segment_dists = np.sqrt(dx**2 + dy**2)
    segment_dists = np.insert(segment_dists, 0, 0)
    return np.cumsum(segment_dists)

def plot_track_ruler():
    try:
        # 1. 讀取檔案
        jsf_files = sorted(glob.glob(os.path.join(JSF_FOLDER, "*.jsf")))
        if not jsf_files: return
        target_jsf = jsf_files[TARGET_FILE_INDEX]
        
        print(f"正在分析檔案: {os.path.basename(target_jsf)}")
        lons, lats = read_jsf_data(target_jsf)
        
        if len(lons) == 0:
            print("錯誤: 無 GPS 資料")
            return

        # 2. 計算距離
        dists = calculate_cumulative_distance(lons, lats)
        total_len = dists[-1]
        
        print("="*40)
        print(f"【測線資訊報告】")
        print(f"總長度: {total_len:.2f} 公尺")
        print(f"刻度間距: 每 {MARKER_INTERVAL} 公尺標示一次")
        print("="*40)

        # 3. 準備底圖
        print("繪製地圖中...")
        df = pd.read_csv(XYZ_FILE, sep=r'\s+', header=None, names=['x', 'y', 'z'])
        trans_bg = Transformer.from_crs(CRS_XYZ, CRS_GPS, always_xy=True)
        bg_lon, bg_lat = trans_bg.transform(df['x'].values, df['y'].values)

        plt.figure(figsize=FIG_SIZE, dpi=DPI)
        
        # 畫底圖
        plt.scatter(bg_lon, bg_lat, c=df['z'], s=0.01, cmap=XYZ_CMAP, alpha=0.6)
        plt.colorbar(label='Depth (m)')
        
        # 畫測線
        plt.plot(lons, lats, color='black', linewidth=1.5, label='Survey Track')

        # --- 4. 畫刻度 (關鍵步驟) ---
        # 產生要標示的距離點: 0, 100, 200, 300... 直到總長
        marker_dists = np.arange(0, total_len, MARKER_INTERVAL)
        
        for m_dist in marker_dists:
            # 找出最接近該距離的索引位置
            idx = (np.abs(dists - m_dist)).argmin()
            
            # 取得該點座標
            mx, my = lons[idx], lats[idx]
            
            # 畫點
            plt.plot(mx, my, 'o', color=MARKER_BG, markersize=4)
            
            # 寫數字 (例如 "100m")
            # bbox 讓文字有個背景框，不怕跟地圖顏色混在一起
            plt.text(mx, my, f"{int(m_dist)}", 
                     color=MARKER_COLOR, fontsize=8, fontweight='bold',
                     ha='right', va='bottom',
                     bbox=dict(boxstyle="round,pad=0.2", fc=MARKER_BG, ec="none", alpha=0.8))

        # 標示終點總長
        plt.text(lons[-1], lats[-1], f"End: {int(total_len)}m", 
                 color='white', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle="square,pad=0.3", fc="black", ec="none"))

        plt.title(f"Trackline Ruler (Total: {total_len:.1f}m)\nFile: {os.path.basename(target_jsf)}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    plot_track_ruler()