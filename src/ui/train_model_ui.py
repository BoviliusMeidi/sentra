import streamlit as st
from src.config import KECAMATAN_LIST, ALL_BANDS, DEFAULT_ALPHAS, DEFAULT_L1_RATIOS, DEFAULT_TEST_RATIOS, DEFAULT_SAMPLES

def apply_training_custom_css():
    st.markdown("""
        <style>
            [data-testid="stHorizontalBlock"] { align-items: stretch; }
            .upload-box { border: 1px dashed #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
        </style>
    """, unsafe_allow_html=True)

def render_internal_scope(is_disabled: bool = False):
    subdistrict_options = ["All Subdistricts"] + KECAMATAN_LIST
    st.selectbox("Select Subdistrict", subdistrict_options, disabled=is_disabled, key="input_kec")

    band_options = ["All Bands"] + ALL_BANDS
    selected_band = st.selectbox("Select Target Band", band_options, disabled=is_disabled, key="input_band")

    if selected_band == "All Bands":
        st.text_input("Base Model Name", placeholder="e.g., ciliwung_base", help="Subdistrict & band suffixes will be appended automatically.", disabled=is_disabled, key="int_model_name")
    else:
        st.text_input("Model Name", placeholder="e.g., ciliwung_model_v1", disabled=is_disabled, key="int_model_name")

def render_hyperparameters(is_disabled: bool = False, prefix: str = "int"):
    col1, col2 = st.columns(2)
    with col1:
        st.slider("Temporal Window Size (W)", 3, 7, 5, disabled=is_disabled, step=2, key=f"{prefix}_w")
        st.multiselect("Max Pixel Samples", options=DEFAULT_SAMPLES, default=DEFAULT_SAMPLES, disabled=is_disabled, key=f"{prefix}_samples")
        st.number_input("Random State (Seed)", min_value=0, value=42, disabled=is_disabled, key=f"{prefix}_seed")
    with col2:
        st.multiselect("Train-Test Split Ratio (%)", options=DEFAULT_TEST_RATIOS, default=DEFAULT_TEST_RATIOS, disabled=is_disabled, key=f"{prefix}_ratios")
        st.multiselect("Alpha (Regularization)", options=DEFAULT_ALPHAS, default=DEFAULT_ALPHAS, disabled=is_disabled, key=f"{prefix}_alphas")
        st.multiselect("L1 Ratio (Lasso/Ridge)", options=DEFAULT_L1_RATIOS, default=DEFAULT_L1_RATIOS, disabled=is_disabled, key=f"{prefix}_l1s")

def render_dynamic_uploaders(window_size: int, is_disabled: bool = False):
    st.markdown("#### Upload Sliding Window Data")
    st.info(f"Window Size (W) = {window_size}. Requires imagery data for {window_size} consecutive historical dates.")

    for i in range(1, window_size + 1):
        with st.expander(f"Historical Date {i}", expanded=(i==1)):
            st.file_uploader(f"Date {i} - SCL (Cloud Mask)", disabled=is_disabled, key=f"up_hist_scl_{i}")
            cols = st.columns(3)
            for j, band in enumerate(ALL_BANDS):
                cols[j % 3].file_uploader(f"Date {i} - {band}", disabled=is_disabled, key=f"up_hist_{i}_{band}")

    st.markdown("#### Upload Ground Truth Data")
    with st.expander("Ground Truth", expanded=True):
        st.file_uploader("Target - SCL (Cloud Mask)", disabled=is_disabled, key="up_gt_scl")
        cols = st.columns(3)
        for j, band in enumerate(ALL_BANDS):
            cols[j % 3].file_uploader(f"Target - {band}", disabled=is_disabled, key=f"up_gt_{band}")

    st.markdown("#### Upload Shapefile Data")
    st.info("Please upload all necessary shapefile components simultaneously (`.shp`, `.shx`, `.dbf`, and `.prj`) to ensure correct spatial coordinate reference system (CRS) binding.")

    shp_col1, shp_col2 = st.columns(2)
    with shp_col1:
        st.file_uploader("Upload (.shp)", type=["shp"], disabled=is_disabled, key="up_shp")
        st.file_uploader("Upload (.shx)", type=["shx"], disabled=is_disabled, key="up_shx")
    with shp_col2:
        st.file_uploader("Upload (.dbf)", type=["dbf"], disabled=is_disabled, key="up_dbf")
        st.file_uploader("Upload (.prj)", type=["prj"], disabled=is_disabled, key="up_prj")