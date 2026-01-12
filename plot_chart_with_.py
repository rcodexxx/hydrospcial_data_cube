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

# 1. 指定第 5 個檔案 (索引從 0 開始，所以是 4)
TARGET_FILE_INDEX = 5

# 2. 設定截取範圍 (630m ~ 780m)
START_DISTANCE = 630.0   # 起始距離
END_DISTANCE   = 780.0   # 結束距離
# 程式會自動計算需要的長度: 150m

# 繪圖設定
FIG_SIZE = (10, 8)
DPI = 120
XYZ_CMAP = 'viridis'     # 底圖顏色
CRS_XYZ = "epsg:3826"    # TWD97
CRS_GPS = "epsg:4326"    # WGS84
# ===========================================

def read_jsf_data(jsf_path):
    """ 讀取 JSF 並回傳經度、緯度、以及原始封包索引 """
    lons, lats, indices = [], [], []
    count = 0
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
                            indices.append(count)
                        elif units == 1:
                            lons.append(rx)
                            lats.append(ry)
                            indices.append(count)
                    
                    count += 1
                    f.seek(size - 240, 1)
                else:
                    f.seek(size, 1)
    except: pass
    return np.array(lons), np.array(lats), np.array(indices)

def calculate_cumulative_distance(lons, lats):
    """ 計算累積距離 """
    transformer = Transformer.from_crs(CRS_GPS, CRS_XYZ, always_xy=True)
    xx, yy = transformer.transform(lons, lats)
    
    dx = np.diff(xx)
    dy = np.diff(yy)
    segment_dists = np.sqrt(dx**2 + dy**2)
    segment_dists = np.insert(segment_dists, 0, 0)
    
    return np.cumsum(segment_dists)

def plot_specific_segment():
    try:
        # --- 1. 準備底圖 ---
        print("1. 讀取地形底圖...")
        df = pd.read_csv(XYZ_FILE, sep=r'\s+', header=None, names=['x', 'y', 'z'])
        trans_bg = Transformer.from_crs(CRS_XYZ, CRS_GPS, always_xy=True)
        bg_lon, bg_lat = trans_bg.transform(df['x'].values, df['y'].values)

        # --- 2. 讀取目標檔案 ---
        jsf_files = sorted(glob.glob(os.path.join(JSF_FOLDER, "*.jsf")))
        if len(jsf_files) <= TARGET_FILE_INDEX:
            print(f"錯誤: 找不到第 {TARGET_FILE_INDEX+1} 個檔案 (總共只有 {len(jsf_files)} 個)")
            return
        
        target_jsf = jsf_files[TARGET_FILE_INDEX]
        print(f"2. 讀取檔案: {os.path.basename(target_jsf)}")
        
        lons, lats, pings = read_jsf_data(target_jsf)
        
        if len(lons) == 0:
            print("錯誤: 無 GPS 資料")
            return

        # --- 3. 計算距離與截取 ---
        dists = calculate_cumulative_distance(lons, lats)
        total_len = dists[-1]
        
        # 截取邏輯：距離在 630 ~ 780 之間
        mask = (dists >= START_DISTANCE) & (dists <= END_DISTANCE)
        
        sel_lons = lons[mask]
        sel_lats = lats[mask]
        sel_pings = pings[mask]
        sel_dists = dists[mask]

        if len(sel_lons) == 0:
            print(f"警告: 選取範圍 ({START_DISTANCE}-{END_DISTANCE}m) 超出測線總長 ({total_len:.1f}m)！")
            return

        # --- 輸出關鍵資訊 (重要！) ---
        print("=" * 40)
        print(f"【截取成功】")
        print(f"檔案: {os.path.basename(target_jsf)}")
        print(f"目標區間: {START_DISTANCE}m ~ {END_DISTANCE}m (長度 {END_DISTANCE-START_DISTANCE}m)")
        print(f"對應 Ping 號碼: #{sel_pings[0]} 到 #{sel_pings[-1]}")
        print(f"資料點數: {len(sel_pings)} 個")
        print("=" * 40)

        # --- 4. 繪圖 ---
        plt.figure(figsize=FIG_SIZE, dpi=DPI)
        
        # A. 底圖
        plt.scatter(bg_lon, bg_lat, c=df['z'], s=0.01, cmap=XYZ_CMAP, alpha=0.5)
        
        # B. 完整測線 (半透明灰色)
        plt.plot(lons, lats, color='gray', linewidth=1, alpha=0.3, label='Full Track')
        
        # C. 目標段落 (顯眼紅色粗線)
        plt.plot(sel_lons, sel_lats, color='red', linewidth=3, label=f'Segment {int(START_DISTANCE)}-{int(END_DISTANCE)}m')
        
        # D. 標示起終點
        plt.plot(sel_lons[0], sel_lats[0], 'o', color='yellow', markeredgecolor='black', markersize=7, label='Start')
        plt.plot(sel_lons[-1], sel_lats[-1], 'X', color='yellow', markeredgecolor='black', markersize=7, label='End')

        # 加文字標籤
        plt.text(sel_lons[0], sel_lats[0], f"{int(START_DISTANCE)}m", color='white', fontweight='bold', bbox=dict(fc='red', ec='none', alpha=0.7))
        plt.text(sel_lons[-1], sel_lats[-1], f"{int(END_DISTANCE)}m", color='white', fontweight='bold', bbox=dict(fc='red', ec='none', alpha=0.7))

        plt.title(f"Selected Segment ({START_DISTANCE}m-{END_DISTANCE}m)\nFile: {os.path.basename(target_jsf)}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    plot_specific_segment()