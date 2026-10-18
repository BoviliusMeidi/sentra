import numpy as np
from src.core.base_data_processor import BaseDataProcessor
from src.data_pipeline.micro_clipper import MicroClipper
from src.config import ALL_BANDS, RANDOM_SEED

class SlidingWindowBuilder(BaseDataProcessor):
    """
    Construct the feature matrix (x) and target vector (y) using a sliding window approach.
    """

    def __init__(self, micro_clipper: MicroClipper):
        self.clipper = micro_clipper;

    def process(self, all_dates: list, W: int, subdist: str, target_band: str, max_pixel_limit: int = 200000) -> tuple:
        target_res = "10m" if target_band in ["B02", "B03", "B04", "B08"] else "20m"
        total_windows = len(all_dates) - W

        temp_X_list, temp_y_list = [], []
        pixels_per_window = max_pixel_limit // total_windows if total_windows > 0 else max_pixel_limit
        rng = np.random.RandomState(RANDOM_SEED)

        for i in range(total_windows):
            window_dates = all_dates[i : i + W]
            target_date = all_dates[i + W]

            y_arr = self.clipper.process(target_date[0], target_date[1], subdist, target_band, target_res)
            if y_arr is None:
                continue

            expected_shape = y_arr.shape[0]
            X_window_list = []
            is_valid_window = True

            for d in window_dates:
                for b in ALL_BANDS:
                    x_arr = self.clipper.process(d[0], d[1], subdist, b, target_res)
                    if x_arr is None or x_arr.shape[0] != expected_shape:
                        is_valid_window = False
                        break
                    X_window_list.append(x_arr)
                if not is_valid_window: break

            if not is_valid_window:
                continue

            X_window_concat = np.column_stack(X_window_list)
            x_valid_mask = ~np.isnan(X_window_concat).any(axis=1)
            final_mask = x_valid_mask & (~np.isnan(y_arr))
            valid_idx = np.where(final_mask)[0]

            if len(valid_idx) > 0:
                chosen_idx = rng.choice(valid_idx, min(len(valid_idx), pixels_per_window), replace=False)
                temp_X_list.append(X_window_concat[chosen_idx])
                temp_y_list.append(y_arr[chosen_idx])

        if not temp_X_list:
            return None, None

        return np.vstack(temp_X_list), np.concatenate(temp_y_list)