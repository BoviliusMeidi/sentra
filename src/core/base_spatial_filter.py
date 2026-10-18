from abc import ABC, abstractmethod
import numpy as np

class BaseSpatialFilter(ABC):
    """
    Abstract contract for 2D Spatial transformations (Clipping, Masking)
    """

    @abstractmethod
    def apply(self, raster_data: np.ndarray, meta: dict, *args, **kwargs) -> np.ndarray:
        pass