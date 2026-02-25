# utils.py
import struct
import os
import numpy as np
import pandas as pd
from pyproj import Transformer

# Coordinate Reference Systems
CRS_TWD97 = "epsg:3826"
CRS_WGS84 = "epsg:4326"

def read_xyz(file_path):
    """Reads standard whitespace-separated XYZ files (TWD97)."""
    return pd.read_csv(file_path, sep=r'\s+', header=None, names=['x', 'y', 'z'])

def read_jsf(file_path):
    """
    Parses EdgeTech JSF files (Msg 80 only).
    Returns list of dicts: {'ping', 'lon', 'lat', 'amps'}.
    """
    pings_data = []
    file_size = os.path.getsize(file_path)
    counter = 0

    try:
        with open(file_path, 'rb') as f:
            while f.tell() < file_size:
                # 1. Read packet header (16 bytes)
                header = f.read(16)
                if len(header) < 16: break
                
                msg_type = struct.unpack_from('<H', header, 4)[0]
                packet_size = struct.unpack_from('<I', header, 12)[0]

                # 2. Process Sonar Data (Msg 80)
                if msg_type == 80 and packet_size >= 240:
                    type80_header = f.read(240)
                    
                    # Extract metadata
                    num = struct.unpack_from('<I', type80_header, 4)[0]
                    rx = struct.unpack_from('<i', type80_header, 80)[0]
                    ry = struct.unpack_from('<i', type80_header, 84)[0]
                    units = struct.unpack_from('<h', type80_header, 88)[0]

                    if num == 0:
                        num = counter

                    counter += 1

                    # Coordinate scaling (Arcseconds to Degrees)
                    lon, lat = 0.0, 0.0
                    if units == 2 and rx != 0:
                        lon = rx / 600000.0
                        lat = ry / 600000.0
                    elif units == 1 and rx != 0:
                        lon = float(rx)
                        lat = float(ry)

                    # 3. Process Payload (Analytic Signal)
                    data_len = packet_size - 240
                    if data_len > 0:
                        raw_bytes = f.read(data_len)
                        raw_data = np.frombuffer(raw_bytes, dtype=np.int16)
                        
                        # Compute Envelope: sqrt(Real^2 + Imag^2)
                        try:
                            complex_data = raw_data.reshape(-1, 2)
                            envelope = np.hypot(complex_data[:, 0], complex_data[:, 1])
                        except ValueError:
                            envelope = np.abs(raw_data) # Fallback

                        pings_data.append({
                            'ping': num,
                            'lon': lon,
                            'lat': lat,
                            'amps': envelope
                        })
                else:
                    # Skip other message types
                    f.seek(packet_size, 1)

    except Exception as e:
        print(f"Error reading JSF: {e}")

    return pings_data

def twd97_to_wgs84(x, y):
    """Transforms TWD97 coordinates to WGS84 (Lon/Lat)."""
    transformer = Transformer.from_crs(CRS_TWD97, CRS_WGS84, always_xy=True)
    return transformer.transform(x, y)

def wgs84_to_twd97(lon, lat):
    """Transforms WGS84 coordinates to TWD97 (X/Y)."""
    transformer = Transformer.from_crs(CRS_WGS84, CRS_TWD97, always_xy=True)
    return transformer.transform(lon, lat)