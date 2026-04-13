import rasterio
from rasterio.enums import Resampling
from pathlib import Path

def build_pyramids(tif_dir):
    # 遍歷資料夾內所有 .tif 檔案
    tif_path_list = Path(tif_dir).glob("*.tif")
    
    factors = [2, 4, 8, 16, 32] # 縮放倍率 (金字塔層級)
    
    for tif_path in tif_path_list:
        print(f"正在處理: {tif_path.name}...")
        
        # 以 'r+' (讀寫) 模式開啟
        with rasterio.open(tif_path, 'r+') as dst:
            # 建立 Overviews
            dst.build_overviews(factors, Resampling.average)
            
            # 更新標籤告訴 rio-tiler 此檔案已有縮圖
            dst.update_tags(ns='rio_overview', resampling='average')
            
    print("✅ 所有圖層縮圖建立完成！")