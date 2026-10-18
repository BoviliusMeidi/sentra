import warnings
import os
import logging
import numpy as np
import geopandas as gpd
import joblib
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
import rioxarray
from rasterio.features import geometry_mask

os.environ["SHAPE_RESTORE_SHX"] = "YES"

import src.config as config
from src.config import ALL_BANDS
from src.data_pipeline.temporal_data_discoverer import TemporalDataDiscoverer
from src.data_pipeline.oot_isolator import OutOftimeIsolator
from src.data_pipeline.micro_clipper import MicroClipper
from src.data_pipeline.vector_clipper import VectorClipper
from src.data_pipeline.scl_cloud_masker import SclCloudMasker
from src.data_pipeline.raster_utils import extract_pixels
from src.models.spatial_evaluator import SpatialEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

class EvaluateModelOrchestrator:
    """
    Central orchestrator for Object-Oriented spatial evaluation and Out-of-Time (OOT) validation.
    Handles metric computation and spatial reconstructions for internal database quarantine and custom uploads.
    """
    def __init__(self):
        self.discoverer = TemporalDataDiscoverer()
        self.clipper_instance = MicroClipper()
        self.cloud_masker = SclCloudMasker()
        self.spatial_evaluator = SpatialEvaluator()

    def _generate_rgb_composite(self, dict_of_images: dict):
        b04 = dict_of_images.get("B04")
        b03 = dict_of_images.get("B03")
        b02 = dict_of_images.get("B02")

        if b04 is not None and b03 is not None and b02 is not None:
            rgb = np.dstack((b04, b03, b02))
            valid_pixels = rgb[~np.isnan(rgb).any(axis=2)]

            if valid_pixels.size > 0:
                p2, p98 = np.percentile(valid_pixels, (2, 98), axis=0)

                p98 = np.clip(p98, a_min=1000.0, a_max=3500.0)
                p2 = np.clip(p2, a_min=0.0, a_max=500.0)

                p98[p98 == p2] = p2[p98 == p2] + 1.0
                rgb_stretched = np.clip((rgb - p2) / (p98 - p2) * 255.0, 0, 255)
            else:
                rgb_stretched = np.clip(rgb / 3000.0 * 255.0, 0, 255)

            nan_mask = np.isnan(rgb).any(axis=2)
            rgb_uint8 = np.nan_to_num(rgb_stretched, nan=0.0).astype(np.uint8)
            alpha_uint8 = np.where(nan_mask, 0, 255).astype(np.uint8)

            return np.dstack((rgb_uint8, alpha_uint8))
        return None

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

    def execute_single_band_evaluation(self, district: str, band: str, model_base_name: str, window_size: int = 5, target_date_str: str = None):
        logging.info(f"🔬 Initializing Out-of-Time (OOT) evaluation protocol for district: {district}, band: {band}...")
        band_model_dir = config.SAVED_MODELS_DIR / district / band

        target_filename = f"{model_base_name}_{band}.pkl"
        fallback_filename = f"{model_base_name}.pkl"
        saved_model_path = band_model_dir / target_filename

        if not saved_model_path.exists():
            saved_model_path = band_model_dir / fallback_filename

        if not saved_model_path.exists():
            logging.error(f"❌ Model binary not found at {saved_model_path}")
            return {"success": False, "error": f"Model artifact {model_base_name} was not located in directory {band_model_dir}."}

        elastic_net_model = joblib.load(saved_model_path)
        logging.info(f"✅ Successfully loaded serialized Elastic Net model artifact: {saved_model_path.name}")

        all_dates = self.discoverer.process(config.MICRO_ROOT, district)
        isolator = OutOftimeIsolator(all_dates, window_size, district, band, self.clipper_instance)

        if target_date_str and target_date_str not in ["Auto-Detect", "N/A"]:
            _, target_date_string = isolator.isolate_sequence_from_metadata(target_date_str)
        else:
            _, target_date_string = isolator.isolate_training_sequence()

        sorted_tuples = sorted(all_dates, key=lambda x: x[1])
        target_idx = next(i for i, v in enumerate(sorted_tuples) if v[1] == target_date_string)

        historical_dates_sequence = sorted_tuples[target_idx - window_size : target_idx]
        target_date_tuple = sorted_tuples[target_idx]
        logging.info(f"📅 Quarantined validation sequence established -> Target Ground Truth: {target_date_string} with historical window W={window_size}.")

        spatial_resolution = "10m" if band in ["B02", "B03", "B04", "B08"] else "20m"
        shapefile_path = os.path.join(config.SHP_KECAMATAN_DIR, f"{district}.shp")
        district_shapefile = gpd.read_file(shapefile_path) if os.path.exists(shapefile_path) else None

        dataset_directory = os.path.join(config.MICRO_ROOT, target_date_tuple[0], target_date_tuple[1], district)
        reference_filename = f"Micro_{district}_{band}_{spatial_resolution}.tif"
        reference_filepath = os.path.join(dataset_directory, spatial_resolution, reference_filename)

        try:
            reference_raster = rioxarray.open_rasterio(reference_filepath)
        except Exception as e:
            logging.error(f"❌ Failed to load reference raster: {e}")
            return {"success": False, "error": f"Target reference raster {reference_filename} is missing for observation date {target_date_string}."}

        reprojected_shapefile = district_shapefile.to_crs(reference_raster.rio.crs)
        clipped_reference_raster = reference_raster.rio.clip(reprojected_shapefile.geometry, reprojected_shapefile.crs, drop=False)
        image_height, image_width = clipped_reference_raster.shape[1], clipped_reference_raster.shape[2]

        historical_features_list = []
        for date_info in historical_dates_sequence:
            for b in ALL_BANDS:
                x_arr = extract_pixels(date_info[0], date_info[1], district, b, target_res=spatial_resolution)
                historical_features_list.append(x_arr)

        historical_feature_matrix = np.column_stack(historical_features_list)

        target_pixel_array = extract_pixels(target_date_tuple[0], target_date_tuple[1], district, band, target_res=spatial_resolution)
        valid_shapefile_mask = ~np.isnan(target_pixel_array)

        raw_filename = f"Raw_{district}_{band}_{spatial_resolution}.tif"
        raw_filepath = os.path.join(dataset_directory, spatial_resolution, raw_filename)

        shapefile_geometry_mask = geometry_mask(
            geometries=reprojected_shapefile.geometry,
            out_shape=(image_height, image_width),
            transform=clipped_reference_raster.rio.transform(),
            invert=True
        )

        if os.path.exists(raw_filepath):
            try:
                raw_raster = rioxarray.open_rasterio(raw_filepath)
                clipped_raw = raw_raster.rio.clip(reprojected_shapefile.geometry, reprojected_shapefile.crs, drop=False)
                raw_2d = clipped_raw.values[0].astype(float)
                raw_2d[raw_2d == 0] = np.nan
                raw_2d[~shapefile_geometry_mask] = np.nan
            except Exception:
                raw_2d = target_pixel_array.reshape(image_height, image_width)
        else:
            raw_2d = target_pixel_array.reshape(image_height, image_width)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            temporal_mean_values = np.nanmean(historical_feature_matrix, axis=1, keepdims=True)

        imputed_feature_matrix = np.where(np.isnan(historical_feature_matrix), temporal_mean_values, historical_feature_matrix)
        spatial_median_values = np.nanmedian(imputed_feature_matrix, axis=0)
        for column_index in range(imputed_feature_matrix.shape[1]):
            imputed_feature_matrix[:, column_index] = np.where(
                np.isnan(imputed_feature_matrix[:, column_index]),
                spatial_median_values[column_index],
                imputed_feature_matrix[:, column_index]
            )

        valid_prediction_mask = ~np.isnan(imputed_feature_matrix).any(axis=1) & valid_shapefile_mask
        full_reconstructed_image = np.full_like(target_pixel_array, np.nan)

        if np.sum(valid_prediction_mask) > 0:
            gap_filled_predictions = elastic_net_model.predict(imputed_feature_matrix[valid_prediction_mask])
            full_reconstructed_image[valid_prediction_mask] = np.clip(gap_filled_predictions, 0.0, 10000.0)

        full_reconstructed_image[~valid_shapefile_mask] = np.nan

        strict_evaluation_mask = ~np.isnan(historical_feature_matrix).any(axis=1) & ~np.isnan(target_pixel_array) & valid_shapefile_mask
        if np.sum(strict_evaluation_mask) == 0:
            logging.warning("⚠️ Insufficient cloud-free spatial observations found across the quarantine temporal window.")
            return {"success": False, "error": "Insufficient cloud-free spatial observations found across the validation temporal window."}

        strict_predictions = elastic_net_model.predict(historical_feature_matrix[strict_evaluation_mask])
        strict_predictions = np.clip(strict_predictions, 0.0, 10000.0)
        strict_ground_truth = target_pixel_array[strict_evaluation_mask]

        calculated_metrics = self.spatial_evaluator.evaluate(strict_ground_truth, strict_predictions)
        logging.info(f"📊 Evaluation metrics computed successfully for band {band} -> R²: {calculated_metrics.get('R2', 0):.4f}")

        return {
            "success": True,
            "metrics": calculated_metrics,
            "image_ground_truth": raw_2d,
            "image_reconstructed": full_reconstructed_image.reshape(image_height, image_width),
            "target_date_string": target_date_string,
            "band": band
        }

    def execute_all_bands_evaluation(self, district: str, model_base_name: str, window_size: int = 5, target_date_str: str = None):
        logging.info(f"🌐 Executing multispectral evaluation pipeline across all 6 spectral bands for district: {district}...")
        all_metrics = []
        gt_images_2d = {}
        pred_images_2d = {}
        band_raw_results = {}
        target_date_global = None

        for band in ALL_BANDS:
            res = self.execute_single_band_evaluation(district, band, model_base_name, window_size, target_date_str)
            if res["success"]:
                res["metrics"]["Band"] = band
                all_metrics.append(res["metrics"])
                gt_images_2d[band] = res["image_ground_truth"]
                pred_images_2d[band] = res["image_reconstructed"]
                band_raw_results[band] = res
                target_date_global = res["target_date_string"]
            else:
                logging.error(f"❌ Multispectral evaluation halted at band {band}: {res['error']}")
                return {"success": False, "error": f"Evaluation failed for band {band}: {res['error']}"}

        rgb_gt = self._generate_rgb_composite(gt_images_2d)
        rgb_pred = self._generate_rgb_composite(pred_images_2d)
        logging.info("✨ Multispectral composite True Color RGB successfully synthesized.")

        return {
            "success": True,
            "metrics_list": all_metrics,
            "band_raw_results": band_raw_results,
            "target_date_string": target_date_global,
            "rgb_ground_truth": rgb_gt,
            "rgb_reconstructed": rgb_pred
        }

    def execute_custom_single_band_evaluation(self, district: str, band: str, model_base_name: str, hist_payload: list, gt_path: str, gt_scl_path: str, shp_path: str):
        band_model_dir = config.SAVED_MODELS_DIR / district / band
        target_filename = f"{model_base_name}_{band}.pkl"
        fallback_filename = f"{model_base_name}.pkl"
        saved_model_path = band_model_dir / target_filename

        if not saved_model_path.exists():
            saved_model_path = band_model_dir / fallback_filename

        if not saved_model_path.exists():
            return {"success": False, "error": f"Model {model_base_name} was not found in directory {band_model_dir}."}

        elastic_net_model = joblib.load(saved_model_path)
        clipper = VectorClipper(shp_path) if shp_path else None

        try:
            y_arr_2d_raw, y_meta = self._extract_and_preprocess(gt_path, is_scl=False, clipper=clipper)
            y_arr_2d = y_arr_2d_raw.copy()

            if gt_scl_path:
                gt_scl_arr, _ = self._extract_and_preprocess(gt_scl_path, is_scl=True, clipper=clipper, ref_meta=y_meta)
                y_arr_2d = self.cloud_masker.apply(y_arr_2d, gt_scl_arr)
        except Exception as e:
            return {"success": False, "error": f"Failed to extract Ground Truth target: {str(e)}"}

        target_pixel_array = y_arr_2d.flatten()
        valid_shapefile_mask = ~np.isnan(target_pixel_array)

        historical_features_list = []
        try:
            for date_data in hist_payload:
                scl_path = date_data.get("scl")
                scl_arr = None
                if scl_path:
                    scl_arr, _ = self._extract_and_preprocess(scl_path, is_scl=True, clipper=clipper, ref_meta=y_meta)

                for b in ALL_BANDS:
                    b_path = date_data["bands"].get(b)
                    if not b_path:
                        raise ValueError(f"Historical file for band {b} is missing in the temporal window.")

                    x_arr_2d, _ = self._extract_and_preprocess(b_path, is_scl=False, clipper=clipper, ref_meta=y_meta)
                    if scl_arr is not None:
                        x_arr_2d = self.cloud_masker.apply(x_arr_2d, scl_arr)

                    historical_features_list.append(x_arr_2d.flatten())
        except Exception as e:
             return {"success": False, "error": f"Failed to extract historical feature matrix: {str(e)}"}

        historical_feature_matrix = np.column_stack(historical_features_list)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            temporal_mean_values = np.nanmean(historical_feature_matrix, axis=1, keepdims=True)

        imputed_feature_matrix = np.where(np.isnan(historical_feature_matrix), temporal_mean_values, historical_feature_matrix)

        spatial_median_values = np.nanmedian(imputed_feature_matrix, axis=0)
        for column_index in range(imputed_feature_matrix.shape[1]):
            imputed_feature_matrix[:, column_index] = np.where(
                np.isnan(imputed_feature_matrix[:, column_index]),
                spatial_median_values[column_index],
                imputed_feature_matrix[:, column_index]
            )

        valid_prediction_mask = ~np.isnan(imputed_feature_matrix).any(axis=1) & valid_shapefile_mask
        full_reconstructed_image = np.full_like(target_pixel_array, np.nan)

        if np.sum(valid_prediction_mask) > 0:
            gap_filled_predictions = elastic_net_model.predict(imputed_feature_matrix[valid_prediction_mask])
            full_reconstructed_image[valid_prediction_mask] = np.clip(gap_filled_predictions, 0.0, 10000.0)

        full_reconstructed_image[~valid_shapefile_mask] = np.nan

        strict_evaluation_mask = ~np.isnan(historical_feature_matrix).any(axis=1) & ~np.isnan(target_pixel_array) & valid_shapefile_mask
        if np.sum(strict_evaluation_mask) == 0:
            return {"success": False, "error": "Insufficient spatial observations after cloud masking and vector clipping."}

        strict_predictions = elastic_net_model.predict(historical_feature_matrix[strict_evaluation_mask])
        strict_predictions = np.clip(strict_predictions, 0.0, 10000.0)
        strict_ground_truth = target_pixel_array[strict_evaluation_mask]

        calculated_metrics = self.spatial_evaluator.evaluate(strict_ground_truth, strict_predictions)
        image_height, image_width = y_meta["height"], y_meta["width"]

        return {
            "success": True,
            "metrics": calculated_metrics,
            "image_ground_truth": y_arr_2d_raw,
            "image_reconstructed": full_reconstructed_image.reshape(image_height, image_width),
            "target_date_string": "Custom Upload",
            "band": band
        }

    def execute_custom_all_bands_evaluation(self, district: str, model_base_name: str, hist_payload: list, gt_paths: dict, gt_scl_path: str, shp_path: str):
        all_metrics = []
        gt_images_2d = {}
        pred_images_2d = {}
        band_raw_results = {}

        for band in ALL_BANDS:
            gt_path = gt_paths.get(band)
            if not gt_path:
                return {"success": False, "error": f"Ground Truth file for band {band} was not uploaded."}

            res = self.execute_custom_single_band_evaluation(
                district=district, band=band, model_base_name=model_base_name,
                hist_payload=hist_payload, gt_path=gt_path, gt_scl_path=gt_scl_path, shp_path=shp_path
            )

            if res["success"]:
                res["metrics"]["Band"] = band
                all_metrics.append(res["metrics"])
                gt_images_2d[band] = res["image_ground_truth"]
                pred_images_2d[band] = res["image_reconstructed"]
                band_raw_results[band] = res
            else:
                return {"success": False, "error": f"Band {band} evaluation failed: {res['error']}"}

        rgb_gt = self._generate_rgb_composite(gt_images_2d)
        rgb_pred = self._generate_rgb_composite(pred_images_2d)

        return {
            "success": True,
            "metrics_list": all_metrics,
            "band_raw_results": band_raw_results,
            "target_date_string": "Custom Upload",
            "rgb_ground_truth": rgb_gt,
            "rgb_reconstructed": rgb_pred
        }