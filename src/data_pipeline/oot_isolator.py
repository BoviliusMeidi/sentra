import logging
import numpy as np

class OutOftimeIsolator:
    """
    Object-Oriented class designed for Out-of-Time (OOT) Validation Isolation.
    Supports direct extraction of quarantined evaluation sequences utilizing pre-trained model metadata.
    """
    def __init__(self, discoverer_dates: list, window_size: int, subdistrict: str, band: str, micro_clipper=None):
        self.sorted_tuples = sorted(discoverer_dates, key=lambda x: x[1])
        self.window_size = window_size
        self.subdistrict = subdistrict
        self.band = band
        self.micro_clipper = micro_clipper

    def isolate_sequence_from_metadata(self, target_date_str: str) -> tuple:
        """
        Isolates the training and evaluation sequences directly from the encapsulated
        model metadata OOT target date, ensuring exact alignment with the training footprint.
        """
        required_len = self.window_size + 1
        if len(self.sorted_tuples) <= required_len:
            raise ValueError(f"Total available temporal records ({len(self.sorted_tuples)}) are insufficient to support Window Size W={self.window_size} plus quarantine requirements.")

        logging.info(f"🔍 OOT Isolator: Locating encapsulated metadata target date '{target_date_str}' for district: {self.subdistrict}, band: {self.band}...")

        try:
            target_idx = next(i for i, v in enumerate(self.sorted_tuples) if v[1] == target_date_str)
        except StopIteration:
            raise ValueError(f"Encapsulated OOT target date '{target_date_str}' from model metadata could not be found in the current temporal catalog.")

        quarantine_start_idx = target_idx - self.window_size
        if quarantine_start_idx < 0:
            raise IndexError(f"Historical temporal window W={self.window_size} exceeds available past records preceding target date {target_date_str}.")

        quarantine_tuples = self.sorted_tuples[quarantine_start_idx:target_idx + 1]
        quarantine_dates = [t[1] for t in quarantine_tuples]

        training_tuples = self.sorted_tuples[:quarantine_start_idx]

        logging.info(f"🎯 OOT Isolator -> Retrieved Encapsulated GT Date: {target_date_str}")
        logging.info(f"🛑 Quarantined for Out-of-Time Evaluation (GT + W History): {quarantine_dates}")
        logging.info(f"📚 Allocated for Training Pipeline: {len(training_tuples)} temporal dates")

        return training_tuples, target_date_str

    def isolate_training_sequence(self) -> tuple:
        """
        Legacy fallback method scanning observation dates for highest valid pixel count
        if model metadata is unavailable.
        """
        if self.micro_clipper is None:
            raise RuntimeError("MicroClipper instance is required for dynamic pixel scanning fallback.")

        valid_indices = range(self.window_size, len(self.sorted_tuples))
        descending_indices = sorted(valid_indices, reverse=True)

        best_tuple = None
        max_valid_pixels = -1
        best_idx = -1

        for idx in descending_indices:
            y_str, d_str = self.sorted_tuples[idx]
            try:
                target_res = "10m" if self.band in ["B02", "B03", "B04", "B08"] else "20m"
                arr = self.micro_clipper.process(y_str, d_str, self.subdistrict, self.band, target_res)
                valid_pixels = int(np.count_nonzero(~np.isnan(arr))) if arr is not None else 0
            except Exception:
                valid_pixels = 0

            if valid_pixels > max_valid_pixels:
                max_valid_pixels = valid_pixels
                best_tuple = self.sorted_tuples[idx]
                best_idx = idx

        target_gt_date = best_tuple[1]
        quarantine_start_idx = best_idx - self.window_size
        training_tuples = self.sorted_tuples[:quarantine_start_idx]

        return training_tuples, target_gt_date