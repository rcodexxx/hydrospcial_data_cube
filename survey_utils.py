# survey_utils.py
import struct
import numpy as np
import pandas as pd
from pyproj import Transformer

# 預設座標系統常數
CRS_TWD97 = "epsg:3826"
CRS_WGS84 = "epsg:4326"

def read_xyz(file_path):
    """讀取多音束 XYZ 資料 (TWD97)"""
    # 使用 pandas 高速讀取
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['x', 'y', 'z'])
    return df

def read_jsf_track(file_path):
    """讀取 JSF 檔案並回傳 (lons, lats, pings)"""
    lons, lats, pings = [], [], []
    count = 0
    try:
        with open(file_path, 'rb') as f:
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
                            pings.append(count)
                        elif units == 1:
                            lons.append(rx)
                            lats.append(ry)
                            pings.append(count)
                    count += 1
                    f.seek(size - 240, 1)
                else:
                    f.seek(size, 1)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        
    return np.array(lons), np.array(lats), np.array(pings)

def twd97_to_wgs84(x, y):
    """將 TWD97 (XY) 轉為 WGS84 (Lon/Lat)"""
    transformer = Transformer.from_crs(CRS_TWD97, CRS_WGS84, always_xy=True)
    return transformer.transform(x, y)

def wgs84_to_twd97(lon, lat):
    """將 WGS84 (Lon/Lat) 轉為 TWD97 (XY)"""
    transformer = Transformer.from_crs(CRS_WGS84, CRS_TWD97, always_xy=True)
    return transformer.transform(lon, lat)

def calc_cumulative_dist(lons, lats):
    """計算路徑累積距離 (回傳單位: 公尺)"""
    # 先轉成投影座標 (公尺) 才能算距離
    xx, yy = wgs84_to_twd97(lons, lats)
    dx = np.diff(xx)
    dy = np.diff(yy)
    dists = np.sqrt(dx**2 + dy**2)
    dists = np.insert(dists, 0, 0) # 補回起點
    return np.cumsum(dists)