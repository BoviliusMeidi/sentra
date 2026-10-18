import numpy as np
import logging
from src.core.base_spatial_filter import BaseSpatialFilter

class SclCloudMasker(BaseSpatialFilter):
    def __init__(self):
        self.cloud_codes = [3, 8, 9, 10]

    def apply(self, target_band_array: np.ndarray, scl_array: np.ndarray, *args, **kwargs) -> np.ndarray:
        if target_band_array.shape != scl_array.shape:
            logging.error("Shape mismatch between target band and SCL band.")
            raise ValueError("SCL and Target arrays must have the same dimensions.")

        logging.info("Applying SCL Cloud Masking...")

        cloud_mask = np.isin(scl_array, self.cloud_codes)

        cleaned_array = target_band_array.astype(float).copy()
        cleaned_array[cloud_mask] = np.nan

        return cleaned_array