import os
import requests
import numpy as np
from typing import Tuple

FIRMS_API_KEY = os.getenv("FIRMS_API_KEY", "YOUR_NASA_FIRMS_API_KEY") 
# Get an API key at https://firms.modaps.eosdis.nasa.gov/api/
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# The dataset we trained on has 12 channels.
# INPUT_FEATURES = [
#     'elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph', 
#     'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask'
# ]
PATCH_SIZE = 64
N_FEATURES = 12

def fetch_firms_data(min_lon: float, min_lat: float, max_lon: float, max_lat: float, day_range: int = 1) -> np.ndarray:
    """
    Fetches VIIRS active fire data from NASA FIRMS for a given bounding box.
    Maps the point anomalies to a 64x64 binary fire grid.
    """
    # Create the bounding box string: [west, south, east, north] -> [min_lon, min_lat, max_lon, max_lat]
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    
    # Using VIIRS SNPP (source ID: VIIRS_SNPP_NRT)
    source = "VIIRS_SNPP_NRT"
    url = f"{FIRMS_BASE_URL}/{FIRMS_API_KEY}/{source}/{bbox_str}/{day_range}"
    
    fire_mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    
    if FIRMS_API_KEY == "YOUR_NASA_FIRMS_API_KEY":
        # Fallback for demo without valid API key: return an empty mask or a simulated random fire
        # For demo purposes, we randomly simulate 5 small fires if no key is provided.
        for _ in range(5):
            rx, ry = np.random.randint(5, PATCH_SIZE - 5, 2)
            fire_mask[rx:rx+3, ry:ry+3] = 1.0
        return fire_mask
        
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse CSV lines
        lines = response.text.strip().split("\n")
        if len(lines) > 1:
            # Header is likely: latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight
            headers = lines[0].split(",")
            lat_idx = headers.index("latitude")
            lon_idx = headers.index("longitude")
            
            # Geographic resolution spans
            lon_span = max_lon - min_lon
            lat_span = max_lat - min_lat
            
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) > lon_idx:
                    lat_val = float(parts[lat_idx])
                    lon_val = float(parts[lon_idx])
                    
                    # Map to 64x64 indices
                    # Lon (X): min_lon -> 0, max_lon -> 63
                    # Lat (Y): max_lat -> 0, min_lat -> 63 (since 0,0 is usually top-left, but we'll map intuitively)
                    x_idx = int(((lon_val - min_lon) / lon_span) * (PATCH_SIZE - 1))
                    y_idx = int(((max_lat - lat_val) / lat_span) * (PATCH_SIZE - 1))
                    
                    if 0 <= x_idx < PATCH_SIZE and 0 <= y_idx < PATCH_SIZE:
                        fire_mask[y_idx, x_idx] = 1.0 # Fire exists here
    except Exception as e:
        print(f"Failed to fetch FIRMS data: {e}")
        
    return fire_mask


def generate_mock_tensor(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> np.ndarray:
    """
    Generates a full 64x64x12 mock tensor, but injects the *real* 
    PrevFireMask fetched from NASA FIRMS into channel 11.
    """
    # Create 64x64x12 tensor filled with baseline normal values (e.g. 0.0 after normalization)
    # We will just use random normal to simulate normalized feature data.
    tensor = np.random.randn(PATCH_SIZE, PATCH_SIZE, N_FEATURES).astype(np.float32)
    
    # PrevFireMask is the 12th feature (index 11)
    fire_mask = fetch_firms_data(min_lon, min_lat, max_lon, max_lat)
    
    tensor[:, :, 11] = fire_mask
    
    return tensor
