"""
Build isopach (sediment thickness) layer from SBP data using Acoustic Physical Principles.

Workflow:
  1. Extract acoustic envelope via Hilbert Transform.
  2. Track Sub-Bottom Reflector (SBR) using Peak Prominence.
  3. Filter out false Multiples (BS, BTB, BSB) using Sonar Draft geometry.
  4. Compute physical thickness via Two-Way Travel Time (TWT).
  5. Use Spatial Interpolation (griddata) to map thickness, bypassing ML regression
     due to anthropogenic dredging disturbances.
"""
from pathlib import Path
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.signal import find_peaks, hilbert
from scipy.interpolate import griddata
from tqdm import tqdm

from src.data_loader.read_sbp_jsf import read_sbp_jsf
from src.config import SOUND_SPEED, EPSG

# ==========================================
# 路徑與常數設定
# ==========================================
ROOT      = Path(__file__).parent.parent.parent
SBP_PATH  = ROOT / "data/sbp"
MBES_TIF  = ROOT / "outputs/tif/mbes_bathymetry.tif"
OUT_THICK = ROOT / "outputs/tif/sbp_isopach.tif"

NS        = 20480e-9
SDEP      = NS * SOUND_SPEED / 2
THICK_CLIP = (0.1, 3.0)  # 厚度合理範圍 (公尺)
SONAR_DRAFT = 1.04       # 聲納吃水深度 (從測量報告取得，用於計算 BS 假訊號)

# ==========================================
# 聲學處理函數
# ==========================================
def find_b(amps, blanking=50):
    """鎖定一次海床 (Sea Floor) 位置"""
    return int(np.argmax(amps[blanking:])) + blanking

def extract_valid_thickness(amps, idx_b):
    """
    結合包絡線與防雷邏輯的厚度萃取核心
    """
    s = idx_b + int(0.15 / SDEP)  # 盲區 15cm
    e = min(len(amps), idx_b + int(4.0 / SDEP)) # 探測深度 4m
    
    if e - s < 10:
        return None

    # 1. 包絡線萃取
    window = amps[s:e].astype(np.float64)
    envelope = np.abs(hilbert(window))
    db_win = 20 * np.log10(np.maximum(envelope, 1.0))
    noise_floor = np.median(db_win) + 5.0

    # 2. 尋找 SBR
    peaks, properties = find_peaks(
        db_win, 
        prominence=3.0, 
        distance=int(0.15 / SDEP),
        height=noise_floor,
        width=2
    )

    if len(peaks) == 0:
        return None

    best_idx = np.argmax(properties["prominences"] * properties["peak_heights"])
    best_peak_pos = peaks[best_idx]
    
    # 3. 計算實體厚度
    thick = (s + best_peak_pos - idx_b) * SDEP 

    # 4. 防雷過濾：檢查是否撞到 BS (Surface-Bottom) 多次反射波
    # 如果找到的厚度與吃水深度極度接近 (誤差 < 15cm)，則視為假訊號並捨棄
    if abs(thick - SONAR_DRAFT) < 0.15:
        return None

    return thick

# ==========================================
# 主程式
# ==========================================
def main():
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)

    # 1. 萃取所有測線的有效厚度
    all_x, all_y, all_thick = [], [], []
    n_total, n_valid, n_rejected = 0, 0, 0

    print("1. Extracting Acoustic Thickness along tracklines...")
    for jsf in tqdm(sorted(SBP_PATH.glob("*.jsf")), desc="Processing JSF"):
        data = read_sbp_jsf(jsf)
        if "SBP" not in data: continue
        
        d = data["SBP"]
        valid = ~np.isnan(d["lon"])
        if valid.sum() == 0: continue
            
        x, y = transformer.transform(d["lon"][valid], d["lat"][valid])
        amps = d["amps"][valid].astype(np.float32)

        for i in range(len(amps)):
            n_total += 1
            idx_b = find_b(amps[i])
            thick = extract_valid_thickness(amps[i], idx_b)
            
            if thick is not None and THICK_CLIP[0] <= thick <= THICK_CLIP[1]:
                all_x.append(x[i])
                all_y.append(y[i])
                all_thick.append(thick)
                n_valid += 1
            else:
                n_rejected += 1

    all_x = np.array(all_x, dtype=np.float64)
    all_y = np.array(all_y, dtype=np.float64)
    all_thick = np.array(all_thick, dtype=np.float64)

    print(f"\nExtraction Summary:")
    print(f"  Total Pings: {n_total}")
    print(f"  Valid SBR Detected: {n_valid} ({100*n_valid/n_total:.1f}%)")
    print(f"  Thickness Range: {all_thick.min():.2f}m ~ {all_thick.max():.2f}m")
    print(f"  Median Thickness: {np.median(all_thick):.2f}m")

    if n_valid < 100:
        print("Error: Not enough valid points for interpolation.")
        return

    # 2. 讀取 MBES 網格作為底圖座標基礎
    print("\n2. Loading Target MBES Grid Geometry...")
    with rasterio.open(MBES_TIF) as src:
        profile = src.profile.copy()
        transform = src.transform
        height, width = src.height, src.width
        mbes_data = src.read(1)
        mbes_nodata = src.nodata

    res = transform.a
    xs_grid = transform.c + (np.arange(width) + 0.5) * res
    ys_grid = transform.f + (np.arange(height) + 0.5) * (-res)
    grid_x, grid_y = np.meshgrid(xs_grid, ys_grid)
    
    # 建立需要內插的有效網格點 (只在有水深資料的地方算厚度)
    if mbes_nodata is not None:
        valid_grid = (mbes_data != mbes_nodata)
    else:
        valid_grid = np.isfinite(mbes_data)

    target_x = grid_x[valid_grid]
    target_y = grid_y[valid_grid]

    print("\n3. Performing Spatial Interpolation (IDW via KDTree)...")
    from scipy.spatial import cKDTree
    
    dx = np.diff(all_x)
    dy = np.diff(all_y)
    dist = np.hypot(dx, dy)
    ping_spacing = np.median(dist[dist > 0])

    MAX_GAP = 70
    max_distance = MAX_GAP * 1.5 
    k_neighbors = 200
    smoothing = MAX_GAP * 2

    points = np.column_stack([all_x, all_y])
    tree = cKDTree(points)

    target_points = np.column_stack([target_x, target_y])
    distances, indices = tree.query(target_points, k=k_neighbors, distance_upper_bound=max_distance)

    interpolated_thick = np.full(len(target_points), np.nan, dtype=np.float32)

    for i in range(len(target_points)):
        valid_mask = np.isfinite(distances[i]) & (distances[i] <= max_distance)
        if not np.any(valid_mask):
            continue 
            
        valid_dist = distances[i][valid_mask]
        valid_idx = indices[i][valid_mask]
        
        weights = 1.0 / (valid_dist ** 2 + smoothing)
        interpolated_thick[i] = np.sum(weights * all_thick[valid_idx]) / np.sum(weights)
    thick_2d = np.full((height, width), -9999.0, dtype=np.float32)
    valid_interp_mask = np.isfinite(interpolated_thick)
    thick_2d[valid_grid] = np.where(valid_interp_mask, interpolated_thick, -9999.0)

    # 4. 輸出 GeoTIFF
    print("\n4. Saving Isopach GeoTIFF...")
    profile.update(dtype="float32", count=1, nodata=-9999.0)
    
    with rasterio.open(OUT_THICK, "w", **profile) as dst:
        dst.write(thick_2d, 1)
        
    print(f"Successfully saved to: {OUT_THICK}")

if __name__ == "__main__":
    main()