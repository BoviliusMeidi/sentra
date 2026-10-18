import os
import warnings
import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
import rioxarray
import xarray as xr
import joblib
from io import BytesIO
import geopandas as gpd
from rasterio.features import geometry_mask

os.environ["SHAPE_RESTORE_SHX"] = "YES"

import src.config as config
from src.config import ALL_BANDS
from src.data_pipeline.vector_clipper import VectorClipper
from src.data_pipeline.scl_cloud_masker import SclCloudMasker
from src.data_pipeline.raster_utils import extract_pixels
from src.ui.image_reconstruction_ui import generate_grayscale_rgba

class ImageReconstructionOrchestrator:
    """
    Central orchestrator for Object-Oriented spatial image reconstruction.
    Handles inference for Internal Database (Gap-Filling & Scratch Reconstruction) and Custom User Uploads.
    """
    def __init__(self):
        self.cloud_masker = SclCloudMasker()

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

    def execute_internal_reconstruction(self, district: str, band: str, model_base_name: str, target_date_str: str, mode: str, window_size: int = 5):
        """Executes internal database reconstruction (Gap-Filling or Scratch reconstruction for missing dates)."""
        band_model_dir = config.SAVED_MODELS_DIR / district / band
        target_filename = f"{model_base_name}_{band}.pkl"
        fallback_filename = f"{model_base_name}.pkl"
        saved_model_path = band_model_dir / target_filename

        if not saved_model_path.exists():
            saved_model_path = band_model_dir / fallback_filename
        if not saved_model_path.exists():
            return {"success": False, "error": f"Model {model_base_name} was not found for band {band}."}

        elastic_net_model = joblib.load(saved_model_path)

        from src.data_pipeline.temporal_data_discoverer import TemporalDataDiscoverer
        discoverer = TemporalDataDiscoverer()
        all_dates = discoverer.process(config.MICRO_ROOT, district)
        sorted_tuples = sorted(all_dates, key=lambda x: x[1])

        spatial_res = "10m" if band in ["B02", "B03", "B04", "B08"] else "20m"
        shapefile_path = os.path.join(config.SHP_KECAMATAN_DIR, f"{district}.shp")
        district_shp = gpd.read_file(shapefile_path) if os.path.exists(shapefile_path) else None

        if mode == "gap_fill":
            try:
                target_idx = next(i for i, v in enumerate(sorted_tuples) if v[1] == target_date_str)
            except StopIteration:
                return {"success": False, "error": f"Target date {target_date_str} not found in repository."}
            if target_idx < window_size:
                return {"success": False, "error": f"Insufficient historical records. Requires W={window_size} past images."}
            historical_dates = sorted_tuples[target_idx - window_size : target_idx]
            reference_tuple = sorted_tuples[target_idx]
        else:
            past_dates = [d for d in sorted_tuples if d[1] < target_date_str]
            if len(past_dates) < window_size:
                return {"success": False, "error": f"Insufficient historical records prior to {target_date_str}."}
            historical_dates = past_dates[-window_size:]
            reference_tuple = historical_dates[-1]

        dataset_dir = os.path.join(config.MICRO_ROOT, reference_tuple[0], reference_tuple[1], district)
        ref_path = os.path.join(dataset_dir, spatial_res, f"Micro_{district}_{band}_{spatial_res}.tif")
        if not os.path.exists(ref_path):
            return {"success": False, "error": f"Reference raster missing."}

        reference_raster = rioxarray.open_rasterio(ref_path)
        reprojected_shp = district_shp.to_crs(reference_raster.rio.crs)
        clipped_ref = reference_raster.rio.clip(reprojected_shp.geometry, reprojected_shp.crs, drop=False)
        img_h, img_w = clipped_ref.shape[1], clipped_ref.shape[2]

        historical_features = []
        for d_info in historical_dates:
            for b in ALL_BANDS:
                x_arr = extract_pixels(d_info[0], d_info[1], district, b, target_res=spatial_res)
                historical_features.append(x_arr)

        hist_matrix = np.column_stack(historical_features)

        shapefile_geometry_mask = geometry_mask(
            geometries=reprojected_shp.geometry,
            out_shape=(img_h, img_w),
            transform=clipped_ref.rio.transform(),
            invert=True
        )

        if mode == "gap_fill":
            target_arr = extract_pixels(reference_tuple[0], reference_tuple[1], district, band, target_res=spatial_res)

            raw_filename = f"Raw_{district}_{band}_{spatial_res}.tif"
            raw_filepath = os.path.join(dataset_dir, spatial_res, raw_filename)
            if os.path.exists(raw_filepath):
                try:
                    raw_raster = rioxarray.open_rasterio(raw_filepath)
                    clipped_raw = raw_raster.rio.clip(reprojected_shp.geometry, reprojected_shp.crs, drop=False)
                    raw_2d = clipped_raw.values[0].astype(float)
                    raw_2d[raw_2d == 0] = np.nan
                except Exception:
                    raw_2d = target_arr.reshape(img_h, img_w)
            else:
                raw_2d = target_arr.reshape(img_h, img_w)
        else:
            target_arr = np.full((img_h * img_w,), np.nan)
            raw_2d = np.full((img_h, img_w), np.nan)

        orig_2d = target_arr.reshape(img_h, img_w)
        orig_2d_vis = raw_2d.reshape(img_h, img_w)

        orig_2d_vis[~shapefile_geometry_mask] = np.nan
        orig_2d[~shapefile_geometry_mask] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            temp_means = np.nanmean(hist_matrix, axis=1, keepdims=True)

        imputed_matrix = np.where(np.isnan(hist_matrix), temp_means, hist_matrix)
        spatial_medians = np.nanmedian(imputed_matrix, axis=0)
        for col_idx in range(imputed_matrix.shape[1]):
            imputed_matrix[:, col_idx] = np.where(np.isnan(imputed_matrix[:, col_idx]), spatial_medians[col_idx], imputed_matrix[:, col_idx])

        pred_arr = np.full((img_h * img_w,), np.nan)
        prediction_mask = ~np.isnan(imputed_matrix).any(axis=1)

        if np.sum(prediction_mask) > 0:
            preds = elastic_net_model.predict(imputed_matrix[prediction_mask])
            pred_arr[prediction_mask] = np.clip(preds, 0.0, 10000.0)

        pred_2d = pred_arr.reshape(img_h, img_w)

        patched_2d = np.where(np.isnan(orig_2d), pred_2d, orig_2d)
        patched_2d[~shapefile_geometry_mask] = np.nan

        patched_da = xr.DataArray(patched_2d.reshape(1, img_h, img_w), coords={"band": [1], "y": clipped_ref.y, "x": clipped_ref.x}, dims=("band", "y", "x"))
        patched_da.rio.write_crs(clipped_ref.rio.crs, inplace=True)
        patched_da.rio.write_nodata(np.nan, inplace=True)

        tiff_buf = BytesIO()
        patched_da.rio.to_raster(tiff_buf, driver="GTiff")

        latlon_orig = xr.DataArray(orig_2d_vis.reshape(1, img_h, img_w), coords={"band": [1], "y": clipped_ref.y, "x": clipped_ref.x}, dims=("band", "y", "x")).rio.write_crs(clipped_ref.rio.crs).rio.reproject("EPSG:4326")
        latlon_patched = patched_da.rio.reproject("EPSG:4326")
        bounds = [[latlon_orig.rio.bounds()[1], latlon_orig.rio.bounds()[0]], [latlon_orig.rio.bounds()[3], latlon_orig.rio.bounds()[2]]]

        rgba_orig = generate_grayscale_rgba(latlon_orig.values[0])
        rgba_patched = generate_grayscale_rgba(latlon_patched.values[0])

        return {
            "success": True,
            "tif_bytes": tiff_buf.getvalue(),
            "spatial_bounds": bounds,
            "rgba_original": rgba_orig,
            "rgba_reconstructed": rgba_patched,
            "orig_array": latlon_orig.values[0],
            "patched_array": latlon_patched.values[0],
            "is_synthetic": True if mode == "scratch" else False
        }

    def execute_custom_reconstruction(self, district: str, band: str, model_base_name: str, hist_payload: list, shp_path: str, target_path: str = None, target_scl_path: str = None):
        """Executes custom user uploads. Updated to allow target image processing."""
        band_model_dir = config.SAVED_MODELS_DIR / district / band
        target_filename = f"{model_base_name}_{band}.pkl"
        fallback_filename = f"{model_base_name}.pkl"
        saved_model_path = band_model_dir / target_filename

        if not saved_model_path.exists():
            saved_model_path = band_model_dir / fallback_filename
        if not saved_model_path.exists():
            return {"success": False, "error": f"Model {model_base_name} was not found for band {band}."}

        elastic_net_model = joblib.load(saved_model_path)
        clipper = VectorClipper(shp_path) if shp_path else None

        historical_features = []
        ref_meta = None
        try:
            for d_data in hist_payload:
                scl_p = d_data.get("scl")
                scl_a = None
                if scl_p:
                    scl_a, ref_meta = self._extract_and_preprocess(scl_p, is_scl=True, clipper=clipper, ref_meta=ref_meta)

                for b in ALL_BANDS:
                    b_p = d_data["bands"].get(b)
                    if not b_p:
                        raise ValueError(f"Historical band file {b} is missing.")
                    x_arr, ref_meta = self._extract_and_preprocess(b_path=b_p, is_scl=False, clipper=clipper, ref_meta=ref_meta)
                    if scl_a is not None:
                        x_arr = self.cloud_masker.apply(x_arr, scl_a)
                    historical_features.append(x_arr.flatten())
        except Exception as e:
            return {"success": False, "error": f"Failed to extract historical features: {str(e)}"}

        hist_matrix = np.column_stack(historical_features)
        img_h, img_w = ref_meta["height"], ref_meta["width"]

        orig_2d = np.full((img_h, img_w), np.nan)
        orig_2d_vis = np.full((img_h, img_w), np.nan)

        if target_path:
            try:
                y_arr_2d_raw, _ = self._extract_and_preprocess(target_path, is_scl=False, clipper=clipper, ref_meta=ref_meta)
                y_arr_2d = y_arr_2d_raw.copy()
                if target_scl_path:
                    scl_arr, _ = self._extract_and_preprocess(target_scl_path, is_scl=True, clipper=clipper, ref_meta=ref_meta)
                    y_arr_2d = self.cloud_masker.apply(y_arr_2d, scl_arr)

                orig_2d_vis = y_arr_2d_raw
                orig_2d = y_arr_2d
            except Exception as e:
                return {"success": False, "error": f"Failed to process target raster: {str(e)}"}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            temp_means = np.nanmean(hist_matrix, axis=1, keepdims=True)

        imputed_matrix = np.where(np.isnan(hist_matrix), temp_means, hist_matrix)
        spatial_medians = np.nanmedian(imputed_matrix, axis=0)
        for col_idx in range(imputed_matrix.shape[1]):
            imputed_matrix[:, col_idx] = np.where(np.isnan(imputed_matrix[:, col_idx]), spatial_medians[col_idx], imputed_matrix[:, col_idx])

        pred_arr = np.full((img_h * img_w,), np.nan)
        prediction_mask = ~np.isnan(imputed_matrix).any(axis=1)

        if np.sum(prediction_mask) > 0:
            preds = elastic_net_model.predict(imputed_matrix[prediction_mask])
            pred_arr[prediction_mask] = np.clip(preds, 0.0, 10000.0)

        pred_2d = pred_arr.reshape(img_h, img_w)
        patched_2d = np.where(np.isnan(orig_2d), pred_2d, orig_2d)

        transform = ref_meta["transform"]
        crs = ref_meta["crs"]

        patched_da = xr.DataArray(patched_2d.reshape(1, img_h, img_w), dims=("band", "y", "x"))
        patched_da.rio.write_crs(crs, inplace=True)
        patched_da.rio.write_transform(transform, inplace=True)

        tiff_buf = BytesIO()
        patched_da.rio.to_raster(tiff_buf, driver="GTiff")

        latlon_orig = xr.DataArray(orig_2d_vis.reshape(1, img_h, img_w), dims=("band", "y", "x")).rio.write_crs(crs).rio.write_transform(transform).rio.reproject("EPSG:4326")
        latlon_patched = patched_da.rio.reproject("EPSG:4326")
        bounds = [[latlon_patched.rio.bounds()[1], latlon_patched.rio.bounds()[0]], [latlon_patched.rio.bounds()[3], latlon_patched.rio.bounds()[2]]]

        rgba_orig = generate_grayscale_rgba(latlon_orig.values[0])
        rgba_patched = generate_grayscale_rgba(latlon_patched.values[0])

        return {
            "success": True,
            "tif_bytes": tiff_buf.getvalue(),
            "spatial_bounds": bounds,
            "rgba_original": rgba_orig,
            "rgba_reconstructed": rgba_patched,
            "orig_array": latlon_orig.values[0],
            "patched_array": latlon_patched.values[0],
            "is_synthetic": True if not target_path else False
        }