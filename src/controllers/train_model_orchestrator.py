import os
import uuid
import logging
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

os.environ["SHAPE_RESTORE_SHX"] = "YES"

import src.config as config
from src.config import ALL_BANDS, KECAMATAN_LIST
from src.data_pipeline.vector_clipper import VectorClipper
from src.data_pipeline.scl_cloud_masker import SclCloudMasker
from src.data_pipeline.temporal_data_discoverer import TemporalDataDiscoverer
from src.data_pipeline.micro_clipper import MicroClipper
from src.data_pipeline.sliding_window_builder import SlidingWindowBuilder
from src.data_pipeline.oot_isolator import OutOftimeIsolator
from src.models.spatial_evaluator import SpatialEvaluator
from src.models.elastic_net_model import ElasticNetModel
from src.utils.temp_workspace_manager import TempWorkspaceManager
from src.utils.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

class TrainModelOrchestrator:
    """
    Arsitektur pusat untuk mengatur aliran ekstraksi data spasial,
    pelatihan model Elastic Net, dan serialisasi biner ber-metadata.
    """
    def __init__(self):
        self.cloud_masker = SclCloudMasker()
        self.evaluator = SpatialEvaluator()
        self.ml_model = ElasticNetModel()
        self.discoverer = TemporalDataDiscoverer()
        self.micro_clipper = MicroClipper()
        self.window_builder = SlidingWindowBuilder(self.micro_clipper)

    def finalize_and_save_model(self, save_path: str, metadata_payload: dict):
        """Mendelegasikan injeksi metadata enkapsulasi dan penyimpanan .pkl."""
        self.ml_model.metadata = metadata_payload
        self.ml_model.save(save_path)

    @staticmethod
    def _save_multisheet_experiment_log(model_name: str, all_bands_history: dict, location: str):
        """Mengekspor matriks akurasi Grid Search ke format .xlsx."""
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c if c.isalnum() else "_" for c in model_name)
        excel_path = logs_dir / f"{safe_name}.xlsx"

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for band, history_list in all_bands_history.items():
                rows = []
                for entry in history_list:
                    row = {
                        "Tahap_Pengujian": "Spatial Reconstruction Training",
                        "Kecamatan": location,
                        "Band": band,
                        "OOT_Target_Date": entry.get("OOT_Target_Date", "N/A"),
                        "Max_Samples_k": entry["Actual_Samples"],
                        "Train_Test_Ratio": entry["Ratio_Label"],
                        "Random_Seed": entry["Random_Seed"],
                        "Alpha": entry["Alpha"],
                        "L1_Ratio": entry["L1_Ratio"],
                        "Train_R2": entry["Train_R2"],
                        "Val_MSE": entry["Val_MSE"],
                        "Val_RMSE": entry["Val_RMSE"],
                        "Val_R2": entry["Val_R2"],
                        "Val_PSNR": entry["Val_PSNR"],
                        "Val_SSIM": entry["Val_SSIM"],
                        "Fit_Status": entry["Fit_Status"]
                    }
                    rows.append(row)
                pd.DataFrame(rows).to_excel(writer, sheet_name=band, index=False)

    def _extract_and_preprocess(self, file_path: str, is_scl: bool = False, clipper: VectorClipper = None, ref_meta: dict = None) -> tuple:
        with rasterio.open(file_path) as src:
            meta = src.meta.copy()
            if clipper:
                image_array, meta = clipper.apply(src, meta)
                image_array = image_array[0]
            else:
                image_array = src.read(1)

            if ref_meta is not None:
                ref_shape = (ref_meta["height"], ref_meta["width"])
                if image_array.shape != ref_shape:
                    resampling_method = Resampling.nearest if is_scl else Resampling.bilinear
                    resampled_array = np.empty(shape=ref_shape, dtype=image_array.dtype)
                    reproject(
                        source=image_array, destination=resampled_array,
                        src_transform=meta["transform"], src_crs=meta["crs"],
                        dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
                        resampling=resampling_method,
                    )
                    image_array = resampled_array
                    meta = ref_meta.copy()

            nodata_val = meta.get("nodata", src.nodata)
            image_array = image_array.astype(float)
            if nodata_val is not None:
                image_array[image_array == nodata_val] = np.nan
            image_array[image_array == 0] = np.nan
            return image_array, meta

    def extract_internal_data(self, subdistrict: str, band: str, window_size: int = 5):
        logging.info(f"📥 Extracting INTERNAL DATABASE for {subdistrict} - {band}...")
        all_dates = self.discoverer.process(config.MICRO_ROOT, subdistrict)
        isolator = OutOftimeIsolator(all_dates, window_size, subdistrict, band, self.micro_clipper)
        training_tuples, oot_gt_date = isolator.isolate_training_sequence()

        X_full, y_full = self.window_builder.process(
            training_tuples, window_size, subdistrict, band, max_pixel_limit=int(1e9)
        )
        if X_full is None or len(y_full) == 0:
            raise ValueError(f"Insufficient historical data found for {subdistrict} - {band}")
        return X_full, y_full, oot_gt_date

    def extract_custom_data(self, band: str, historical_files: list, ground_truth_file: str, gt_scl_file: str = None, shapefile_path: str = None):
        logging.info(f"📥 Extracting CUSTOM UPLOAD DATA for band {band} (Multispectral W * 6 bands)...")
        clipper = VectorClipper(shapefile_path) if shapefile_path else None

        y_arr_2d, y_meta = self._extract_and_preprocess(ground_truth_file, is_scl=False, clipper=clipper)
        if gt_scl_file:
            gt_scl_arr, _ = self._extract_and_preprocess(gt_scl_file, is_scl=True, clipper=clipper, ref_meta=y_meta)
            y_arr_2d = self.cloud_masker.apply(y_arr_2d, gt_scl_arr)
        y_arr = y_arr_2d.flatten()

        X_list = []
        for hist_data in historical_files:
            scl_p = hist_data.get("scl")
            scl_arr = None
            if scl_p:
                scl_arr, _ = self._extract_and_preprocess(scl_p, is_scl=True, clipper=clipper, ref_meta=y_meta)

            for b in ALL_BANDS:
                b_p = hist_data.get("bands", {}).get(b)

                if not b_p:
                    raise ValueError(f"Historical file for band {b} is missing in custom upload payload.")

                x_arr_2d, _ = self._extract_and_preprocess(b_p, is_scl=False, clipper=clipper, ref_meta=y_meta)
                if scl_arr is not None:
                    x_arr_2d = self.cloud_masker.apply(x_arr_2d, scl_arr)

                X_list.append(x_arr_2d.flatten())

        X_full = np.column_stack(X_list)
        valid_mask = ~np.isnan(X_full).any(axis=1) & ~np.isnan(y_arr)
        X_clean, y_clean = X_full[valid_mask], y_arr[valid_mask]

        if len(y_clean) == 0:
            raise ValueError("No valid pixels left after clipping, masking, and NaN removal in custom upload.")

        logging.info(f"✨ Custom multispectral matrix extracted successfully. Shape: {X_clean.shape}")
        return X_clean, y_clean, Path(ground_truth_file).stem

    def run_training_on_extracted_data(self, X_data, y_data, test_size: float = 0.2, random_state: int = 42, alphas: list = None, l1_ratios: list = None):
        if alphas is None: alphas = [0.001, 0.01, 0.1]
        if l1_ratios is None: l1_ratios = [0.3, 0.5, 0.7]

        X_temp, X_test, y_temp, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=random_state)

        best_params, grid_history = self.ml_model.train(
            X_train=X_train, y_train=y_train, alphas=alphas, l1_ratios=l1_ratios,
            evaluator=self.evaluator, X_val=X_val, y_val=y_val, random_state=random_state
        )
        test_metrics = self.ml_model.test_and_evaluate(X_test, y_test, self.evaluator)
        return best_params, test_metrics, grid_history

    def execute_internal_pipeline(self, selected_kec: str, selected_band: str, base_name: str, config_dict: dict, status_container, progress_bar) -> pd.DataFrame:
        kec_queue = KECAMATAN_LIST if selected_kec == "All Subdistricts" else [selected_kec]
        target_queue = ALL_BANDS if selected_band == "All Bands" else [selected_band]
        results = []
        total_tasks = len(kec_queue) * len(target_queue)
        completed_tasks = 0

        try:
            for kec in kec_queue:
                all_bands_history = {}
                for band in target_queue:
                    status_container.write(f"📥 Extracting Image Data for {kec} - Band {band}...")

                    if selected_kec == "All Subdistricts" and selected_band == "All Bands":
                        current_model_name = f"{base_name}_{kec}_{band}"
                    elif selected_kec == "All Subdistricts":
                        current_model_name = f"{base_name}_{kec}"
                    elif selected_band == "All Bands":
                        current_model_name = f"{base_name}_{band}"
                    else:
                        current_model_name = base_name

                    X_master, y_master, oot_gt_date = self.extract_internal_data(kec, band, window_size=config_dict['w'])
                    total_extracted = len(y_master)

                    band_grid_history = []
                    best_overall_metrics, best_overall_config = {}, {}
                    best_overall_r2 = -float('inf')

                    status_container.write(f"⏳ Running Grid Search for {kec} - {band}...")

                    for sample_val in config_dict['samples_list']:
                        max_samples = None if sample_val == "All" else int(sample_val)
                        actual_samples = total_extracted

                        if max_samples and total_extracted > max_samples:
                            rng = np.random.RandomState(config_dict['seed'])
                            idx_choice = rng.choice(total_extracted, max_samples, replace=False)
                            X_subset, y_subset = X_master[idx_choice], y_master[idx_choice]
                            actual_samples = max_samples
                        else:
                            X_subset, y_subset = X_master, y_master

                        for ratio_val in config_dict['ratios_list']:
                            test_ratio_decimal = 1.0 - (ratio_val / 100.0)
                            ratio_label = f"{ratio_val}:{100 - ratio_val}"

                            best_params, metrics, grid_history = self.run_training_on_extracted_data(
                                X_data=X_subset, y_data=y_subset, test_size=test_ratio_decimal, random_state=config_dict['seed'],
                                alphas=config_dict['alphas'], l1_ratios=config_dict['l1s']
                            )

                            for entry in grid_history:
                                entry["Ratio_Label"] = ratio_label
                                entry["Actual_Samples"] = actual_samples
                                entry["Random_Seed"] = config_dict['seed']
                                entry["OOT_Target_Date"] = oot_gt_date
                                band_grid_history.append(entry)

                            current_r2 = metrics.get("R2", -float('inf'))
                            if current_r2 > best_overall_r2:
                                best_overall_r2 = current_r2
                                best_overall_metrics = metrics
                                best_overall_config = {
                                    "Samples": actual_samples, "Ratio": ratio_label,
                                    "Alpha": best_params.get("alpha"), "L1": best_params.get("l1_ratio"),
                                    "Seed": config_dict['seed']
                                }

                                metadata_payload = {
                                    "Samples": actual_samples, "Train_Ratio": ratio_label,
                                    "Alpha": best_params.get("alpha"), "L1_Ratio": best_params.get("l1_ratio"),
                                    "Seed": config_dict['seed'], "Val_R2": round(best_overall_r2, 4),
                                    "OOT_Target": oot_gt_date
                                }
                                save_path = ModelRegistry.generate_model_path(kec, band, current_model_name)
                                self.finalize_and_save_model(save_path, metadata_payload)
                                logging.info(f"🌟 New best model saved for {kec} - {band} (R2: {best_overall_r2:.4f})")

                    all_bands_history[band] = band_grid_history
                    results.append({
                        "Model Name": current_model_name, "Location": kec, "Band": band,
                        "OOT Target": oot_gt_date, "Samples": best_overall_config.get("Samples", "N/A"),
                        "Train/Test": best_overall_config.get("Ratio", "N/A"), "Alpha": best_overall_config.get("Alpha", "N/A"),
                        "L1 Ratio": best_overall_config.get("L1", "N/A"), "Seed": best_overall_config.get("Seed", "N/A"),
                        "MSE": round(best_overall_metrics.get("MSE", 0), 4), "RMSE": round(best_overall_metrics.get("RMSE", 0), 4),
                        "R2": round(best_overall_metrics.get("R2", 0), 4), "PSNR": round(best_overall_metrics.get("PSNR", 0), 2),
                        "SSIM": round(best_overall_metrics.get("SSIM", 0), 4)
                    })

                    completed_tasks += 1
                    progress_bar.progress(int((completed_tasks / total_tasks) * 100))
                    status_container.write(f"✅ Success training for {kec} - {band} (Best R2: {best_overall_r2:.4f})")

                self._save_multisheet_experiment_log(f"{base_name}_{kec}" if selected_kec == "All Subdistricts" else base_name, all_bands_history, kec)

            return pd.DataFrame(results)
        except Exception as e:
            raise RuntimeError(f"Internal Training Failed: {e}")

    def execute_custom_pipeline(self, base_name: str, config_dict: dict, session_state_ref, status_container, progress_bar) -> pd.DataFrame:
        w = config_dict['w']

        safe_base_name = "".join(c if c.isalnum() else "_" for c in base_name)
        session_id = safe_base_name

        temp_mgr = TempWorkspaceManager(base_temp_dir="data/custom_workspace")
        results = []
        all_bands_history = {}

        try:
            shp_file = session_state_ref.get("up_shp")
            shp_path = temp_mgr.save_uploaded_file(shp_file, session_id, custom_name="shapefile.shp") if shp_file else None

            shx_file = session_state_ref.get("up_shx")
            if shx_file:
                temp_mgr.save_uploaded_file(shx_file, session_id, custom_name="shapefile.shx")

            dbf_file = session_state_ref.get("up_dbf")
            if dbf_file:
                temp_mgr.save_uploaded_file(dbf_file, session_id, custom_name="shapefile.dbf")

            prj_file = session_state_ref.get("up_prj")
            if prj_file:
                temp_mgr.save_uploaded_file(prj_file, session_id, custom_name="shapefile.prj")

            for idx, band in enumerate(ALL_BANDS):
                status_container.write(f"📥 Extracting User Data for {band}...")
                current_model_name = f"{base_name}_{band}"

                gt_path = temp_mgr.save_uploaded_file(session_state_ref[f"up_gt_{band}"], session_id, custom_name=f"gt_{band}.tif")
                gt_scl_file = session_state_ref.get("up_gt_scl")
                gt_scl_path = temp_mgr.save_uploaded_file(gt_scl_file, session_id, custom_name="gt_scl.tif") if gt_scl_file else None

                hist_paths = []
                for i in range(1, w + 1):
                    bands_dict = {}
                    for b in ALL_BANDS:
                        h_file = session_state_ref.get(f"up_hist_{i}_{b}")
                        if h_file:
                            bands_dict[b] = temp_mgr.save_uploaded_file(h_file, session_id, custom_name=f"hist_{i}_{b}.tif")

                    hist_scl_file = session_state_ref.get(f"up_hist_scl_{i}")
                    h_scl_path = temp_mgr.save_uploaded_file(hist_scl_file, session_id, custom_name=f"hist_scl_{i}.tif") if hist_scl_file else None

                    hist_paths.append({"bands": bands_dict, "scl": h_scl_path})

                X_master, y_master, oot_gt_date = self.extract_custom_data(
                    band=band, historical_files=hist_paths, ground_truth_file=gt_path,
                    gt_scl_file=gt_scl_path, shapefile_path=shp_path
                )
                total_extracted = len(y_master)

                band_grid_history = []
                best_overall_metrics, best_overall_config = {}, {}
                best_overall_r2 = -float('inf')

                status_container.write(f"⏳ Running Grid Search Configurations for {band}...")

                for sample_val in config_dict['samples_list']:
                    max_samples = None if sample_val == "All" else int(sample_val)
                    actual_samples = total_extracted

                    if max_samples and total_extracted > max_samples:
                        rng = np.random.RandomState(config_dict['seed'])
                        idx_choice = rng.choice(total_extracted, max_samples, replace=False)
                        X_subset, y_subset = X_master[idx_choice], y_master[idx_choice]
                        actual_samples = max_samples
                    else:
                        X_subset, y_subset = X_master, y_master

                    for ratio_val in config_dict['ratios_list']:
                        test_ratio_decimal = 1.0 - (ratio_val / 100.0)
                        ratio_label = f"{ratio_val}:{100 - ratio_val}"

                        best_params, metrics, grid_history = self.run_training_on_extracted_data(
                            X_data=X_subset, y_data=y_subset, test_size=test_ratio_decimal, random_state=config_dict['seed'],
                            alphas=config_dict['alphas'], l1_ratios=config_dict['l1s']
                        )

                        for entry in grid_history:
                            entry["Ratio_Label"] = ratio_label
                            entry["Actual_Samples"] = actual_samples
                            entry["Random_Seed"] = config_dict['seed']
                            entry["OOT_Target_Date"] = oot_gt_date
                            band_grid_history.append(entry)

                        current_r2 = metrics.get("R2", -float('inf'))
                        if current_r2 > best_overall_r2:
                            best_overall_r2 = current_r2
                            best_overall_metrics = metrics
                            best_overall_config = {
                                "Samples": actual_samples, "Ratio": ratio_label,
                                "Alpha": best_params.get("alpha"), "L1": best_params.get("l1_ratio"),
                                "Seed": config_dict['seed']
                            }

                metadata_payload = {
                    "Samples": actual_samples, "Train_Ratio": ratio_label,
                    "Alpha": best_params.get("alpha"), "L1_Ratio": best_params.get("l1_ratio"),
                    "Seed": config_dict['seed'], "Val_R2": round(best_overall_r2, 4),
                    "OOT_Target": oot_gt_date, "Window_Size": w
                }

                save_path = ModelRegistry.generate_model_path("Custom Upload", band, current_model_name)
                self.finalize_and_save_model(save_path, metadata_payload)
                logging.info(f"🌟 New best model saved for {band} (R2: {best_overall_r2:.4f})")

                all_bands_history[band] = band_grid_history
                results.append({
                    "Model Name": current_model_name, "Location": "Custom Upload", "Band": band,
                    "OOT Target": oot_gt_date, "Samples": best_overall_config.get("Samples", "N/A"),
                    "Train/Test": best_overall_config.get("Ratio", "N/A"), "Alpha": best_overall_config.get("Alpha", "N/A"),
                    "L1 Ratio": best_overall_config.get("L1", "N/A"), "Seed": best_overall_config.get("Seed", "N/A"),
                    "MSE": round(best_overall_metrics.get("MSE", 0), 4), "RMSE": round(best_overall_metrics.get("RMSE", 0), 4),
                    "R2": round(best_overall_metrics.get("R2", 0), 4), "PSNR": round(best_overall_metrics.get("PSNR", 0), 2),
                    "SSIM": round(best_overall_metrics.get("SSIM", 0), 4)
                })
                progress_bar.progress(int(((idx + 1) / 6) * 100))
                status_container.write(f"✅ Success training for {band} (Best R2: {best_overall_r2:.4f})")

            self._save_multisheet_experiment_log(base_name, all_bands_history, "Custom Upload")

            status_container.write("📁 Dataset saved permanently to Custom Workspace.")
            return pd.DataFrame(results)

        except Exception as e:
            if 'session_id' in locals(): temp_mgr.cleanup_session(session_id)
            raise RuntimeError(f"Custom Training Failed: {e}")