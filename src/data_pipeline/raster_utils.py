import os
import numpy as np
import rasterio
import src.config as config

def extract_pixels(year, date, subdist, band, target_res):
    base_dir = os.path.join(config.MICRO_ROOT, str(year), str(date), subdist)
    file_name = f"Micro_{subdist}_{band}_{target_res}.tif"
    file_path = os.path.join(base_dir, target_res, file_name)

    if not os.path.exists(file_path):
        return None

    with rasterio.open(file_path) as src:
        data = src.read(1)
        nodata_val = src.nodata
        pixel_array = data.flatten().astype(float)

        if nodata_val is not None:
            pixel_array[pixel_array == nodata_val] = np.nan
        else:
            pixel_array[pixel_array == 0] = np.nan

    return pixel_array