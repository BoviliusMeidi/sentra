import streamlit as st
import numpy as np
import folium
import zipfile
from io import BytesIO
from streamlit_folium import st_folium
import joblib
import pandas as pd
import src.config as config
from src.config import ALL_BANDS

def apply_reconstruction_custom_css():
    st.markdown(
        """
        <style>
            .block-container { padding-top: 2.5rem !important; padding-left: 2.5rem !important; padding-right: 2.5rem !important; }
            div[data-testid="stMetricValue"] { font-size: 1.5rem !important; color: #0284c7; }
        </style>
    """,
        unsafe_allow_html=True,
    )

def fetch_reconstruction_model_history(district: str, band: str) -> pd.DataFrame:
    records = []
    target_bands = ALL_BANDS if band == "All Bands" else [band]

    for b in target_bands:
        band_model_dir = config.SAVED_MODELS_DIR / district / b
        if not band_model_dir.exists():
            continue

        for pkl_file in band_model_dir.glob("*.pkl"):
            model_full_name = pkl_file.stem
            base_name = model_full_name.replace(f"_{b}", "")

            try:
                loaded_model = joblib.load(pkl_file)
                meta = getattr(loaded_model, "metadata", {})
            except Exception:
                meta = {}

            records.append(
                {
                    "Base_Model": base_name,
                    "Band": b,
                    "Samples": meta.get("Samples", "N/A"),
                    "Split_Ratio": meta.get("Train_Ratio", "N/A"),
                    "Alpha": meta.get("Alpha", "N/A"),
                    "L1_Ratio": meta.get("L1_Ratio", "N/A"),
                    "Val_R2": meta.get("Val_R2", "N/A"),
                    "OOT_Target": meta.get("OOT_Target", "Auto-Detect"),
                }
            )

    df = pd.DataFrame(records)
    if not df.empty and band == "All Bands":
        band_counts = df.groupby("Base_Model")["Band"].nunique()
        valid_base_models = band_counts[band_counts == len(ALL_BANDS)].index
        df = df[df["Base_Model"].isin(valid_base_models)]
        if not df.empty:
            df = df.groupby("Base_Model").first().reset_index()
            df["Band"] = "All 6 Bands"
            df["Samples"] = "Variant per Band"
            df["Alpha"] = "Variant"
            df["L1_Ratio"] = "Variant"
    return df

def render_reconstruction_parameters_card(kecamatan_list, spectral_bands):
    with st.container(border=True):
        st.markdown("##### Target Parameters & Model Selector")

        district_col, band_col = st.columns(2)
        with district_col:
            district_options = kecamatan_list + ["Custom Upload"]
            selected_district = st.selectbox(
                "Select Target District", district_options, key="recon_dist"
            )
        with band_col:
            band_options = ["All Bands"] + spectral_bands
            selected_band = st.selectbox(
                "Select Spectral Band", band_options, key="recon_band"
            )

        available_models = []
        selected_model = None

        if selected_band == "All Bands":
            district_dir = config.SAVED_MODELS_DIR / selected_district
            if district_dir.exists():
                band_base_sets = []
                for b in ALL_BANDS:
                    b_dir = district_dir / b
                    if b_dir.exists():
                        b_models = {
                            f.stem.replace(f"_{b}", "") for f in b_dir.glob("*.pkl")
                        }
                        band_base_sets.append(b_models)
                if band_base_sets:
                    common_base_models = set.intersection(*band_base_sets)
                    available_models = sorted(list(common_base_models))

            if not available_models:
                st.warning(
                    f"No complete 6-band composite models detected within saved_models/{selected_district}/."
                )
            else:
                selected_model = st.selectbox(
                    "Select Base Model for Reconstruction",
                    available_models,
                    key="recon_model_all",
                )
        else:
            st.markdown("###### Trained Model History (Compiled Binaries)")
            history_df = fetch_reconstruction_model_history(
                selected_district, selected_band
            )
            if history_df.empty:
                st.warning(
                    f"No compiled spatial models (.pkl) detected within saved_models/{selected_district}/ for {selected_band}."
                )
            else:
                st.dataframe(history_df, width="stretch", hide_index=True)
                available_models = history_df["Base_Model"].unique().tolist()
                selected_model = st.selectbox(
                    "Select Base Model for Reconstruction",
                    available_models,
                    key="recon_model_single",
                )

    return selected_district, selected_band, selected_model

def render_custom_reconstruction_uploaders(prefix: str = "recon"):
    st.markdown("#### Upload Target Imagery for Reconstruction")
    st.info("You only need to upload the single target date (the cloudy raster imagery) that you wish to gap-fill.")

    st.file_uploader("Cloudy Target - SCL (Cloud Mask) (.tif)", key=f"{prefix}_target_scl")

    cols = st.columns(3)
    for j, band in enumerate(ALL_BANDS):
        cols[j % 3].file_uploader(f"Cloudy Target - {band} (.tif)", type=["tif"], key=f"{prefix}_target_{band}")

    st.markdown("#### Spatial Constraint (Optional)")
    st.info(
        "You may optionally upload a Shapefile bundle if you wish to apply a new clipping boundary. If left empty, the system will automatically inherit the spatial constraint used during your model's training phase."
    )

    shp_col1, shp_col2 = st.columns(2)
    with shp_col1:
        st.file_uploader("Upload (.shp)", type=["shp"], key=f"{prefix}_shp")
        st.file_uploader("Upload (.shx)", type=["shx"], key=f"{prefix}_shx")
    with shp_col2:
        st.file_uploader("Upload (.dbf)", type=["dbf"], key=f"{prefix}_dbf")
        st.file_uploader("Upload (.prj)", type=["prj"], key=f"{prefix}_prj")

def generate_grayscale_rgba(image_array):
    """
    Generates a grayscale RGBA visualization layer equipped with dynamic 2%-98%
    percentile contrast stretching to prevent extreme saturation (washout)
    in high-reflectance spectral bands like NIR (B08) and SWIR (B11, B12).
    """
    valid_pixels = image_array[~np.isnan(image_array)]

    if valid_pixels.size > 0:
        p2, p98 = np.percentile(valid_pixels, (2, 98))
        if p98 == p2:
            p98 = p2 + 1.0

        clipped_image = np.clip((image_array - p2) / (p98 - p2) * 255.0, 0, 255)
    else:
        clipped_image = np.clip(image_array / 3000.0 * 255.0, 0, 255)

    normalized_image = np.nan_to_num(clipped_image, nan=0.0).astype(np.uint8)

    alpha_channel = np.where(np.isnan(image_array), 0, 255).astype(np.uint8)

    return np.dstack((normalized_image, normalized_image, normalized_image, alpha_channel))

def generate_rgb_composite_rgba(red_array, green_array, blue_array):
    rgb = np.dstack((red_array, green_array, blue_array))
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

def render_folium_map(image_rgba_matrix, spatial_boundaries, layer_title):
    center_latitude = (spatial_boundaries[0][0] + spatial_boundaries[1][0]) / 2.0
    center_longitude = (spatial_boundaries[0][1] + spatial_boundaries[1][1]) / 2.0

    folium_map_object = folium.Map(
        location=[center_latitude, center_longitude], zoom_start=13, control_scale=True
    )
    folium.raster_layers.ImageOverlay(
        image=image_rgba_matrix,
        bounds=spatial_boundaries,
        opacity=1.0,
        name=layer_title,
    ).add_to(folium_map_object)
    return folium_map_object

def display_map_in_streamlit(
    folium_map_object, container_height=500, map_key: str = "map"
):
    st_folium(
        folium_map_object,
        height=container_height,
        width="stretch",
        returned_objects=[],
        key=map_key,
    )

def render_reconstruction_results(visual_results, title_metadata, is_rgb=False):
    with st.container(border=True):
        st.markdown(f"##### 🗺️ Image Reconstruction Results: {title_metadata}")

        safe_key_prefix = "".join(c if c.isalnum() else "_" for c in title_metadata)

        is_synthetic = visual_results.get("is_synthetic", False)

        if is_synthetic:
            st.markdown("**Fully Reconstructed Output (Gap-Free Synthesis)**")
            map_obj = render_folium_map(
                visual_results["rgba_reconstructed"],
                visual_results["spatial_bounds"],
                "Reconstructed Synthesis",
            )
            display_map_in_streamlit(map_obj, map_key=f"{safe_key_prefix}_synth")
        else:
            original_col, reconstructed_col = st.columns(2)
            with original_col:
                st.markdown("**Original Target**")
                map_obj1 = render_folium_map(
                    visual_results["rgba_original"],
                    visual_results["spatial_bounds"],
                    "Cloudy Target",
                )
                display_map_in_streamlit(map_obj1, map_key=f"{safe_key_prefix}_orig")
            with reconstructed_col:
                st.markdown("**Gap-Filled Reconstruction**")
                map_obj2 = render_folium_map(
                    visual_results["rgba_reconstructed"],
                    visual_results["spatial_bounds"],
                    "Gap-Filled Output",
                )
                display_map_in_streamlit(map_obj2, map_key=f"{safe_key_prefix}_recon")

def render_download_assets(reconstruction_tif_bytes, selected_district, target_label):
    with st.container(border=True):
        st.markdown("##### 📥 Export Reconstructed GeoTIFFs")
        if len(reconstruction_tif_bytes) == 1:
            band_key = list(reconstruction_tif_bytes.keys())[0]
            st.download_button(
                label=f"⬇️ Download Reconstructed GeoTIFF ({band_key})",
                data=reconstruction_tif_bytes[band_key],
                file_name=f"Reconstructed_{selected_district}_{band_key}_{target_label}.tif",
                mime="image/tiff",
                type="primary",
                width="stretch",
            )
        else:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(
                zip_buffer, "a", zipfile.ZIP_DEFLATED, False
            ) as zip_file:
                for band_id, byte_data in reconstruction_tif_bytes.items():
                    zip_file.writestr(
                        f"Reconstructed_{selected_district}_{band_id}_{target_label}.tif",
                        byte_data,
                    )

            st.download_button(
                label="Download ZIP Archive (All 6 Bands)",
                data=zip_buffer.getvalue(),
                file_name=f"Reconstructed_{selected_district}_{target_label}_AllBands.zip",
                mime="application/zip",
                type="primary",
                width="stretch",
            )
