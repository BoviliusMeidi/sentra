import os
from src.core.base_data_processor import BaseDataProcessor

class TemporalDataDiscoverer(BaseDataProcessor):
    """
    Scan the filesystem to discover available temporal spatial data.
    """
    def __init__(self, start_year: int = 2017, end_year: int = 2026):
        self.start_year = start_year
        self.end_year = end_year

    def process(self, micro_root: str, subdistrict: str) -> list:
        dates_list = []
        for y in range (self.start_year, self.end_year + 1):
            y_dir = os.path.join(micro_root, str(y));
            if os.path.exists(y_dir):
                for d in os.listdir(y_dir):
                    if os.path.isdir(os.path.join(y_dir, d, subdistrict)):
                        dates_list.append((str(y), d))

        return sorted(dates_list, key=lambda x:x[1])