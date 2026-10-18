import streamlit as st
import matplotlib.pyplot as plt
import io
import pandas as pd
import joblib
import src.config as config
from src.config import ALL_BANDS

def apply_evaluate_custom_css():
    st.markdown(
        """
        <style>
            div[data-testid="stMetricValue"] { font-size: 1.5rem !important; color: #0284c7; }
            .block-container { padding-top: 2.5rem !important; padding-left: 2.5rem !important; padding-right: 2.5rem !important; }
        </style>
    """,
        unsafe_allow_html=True,
    )

def fetch_model_history(district: str, band: str) -> pd.DataFrame:
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

def render_target_parameters_card(kecamatan_list, spectral_bands):
    with st.container(border=True):
        st.markdown("##### Target Parameters & Model History")

        district_col, band_col = st.columns(2)
        with district_col:
            selected_district = st.selectbox(
                "Select Target District", kecamatan_list, key="eval_dist"
            )
        with band_col:
            band_options = ["All Bands"] + spectral_bands
            selected_band = st.selectbox(
                "Select Spectral Band", band_options, key="eval_band"
            )

        available_models = []
        selected_model = None

        if selected_band == "All Bands":
            st.markdown("###### Select Base Model for Evaluation")
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
                    f"No complete 6-band composite models were detected within saved_models/{selected_district}/."
                )
            else:
                selected_model = st.selectbox(
                    "Select Base Model for Evaluation",
                    available_models,
                    key="eval_model_name_all",
                )
                st.session_state["current_oot_target"] = "Auto-Detect"
        else:
            st.markdown("###### Trained Model History")
            history_df = fetch_model_history(selected_district, selected_band)
            if history_df.empty:
                st.warning(
                    f"No compiled spatial models (.pkl) were detected within saved_models/{selected_district}/ for {selected_band}."
                )
            else:
                st.dataframe(history_df, width="stretch", hide_index=True)
                available_models = history_df["Base_Model"].unique().tolist()
                selected_model = st.selectbox(
                    "Select Base Model for Evaluation",
                    available_models,
                    key="eval_model_name_single",
                )

                model_filter = history_df[history_df["Base_Model"] == selected_model]
                default_oot = (
                    model_filter["OOT_Target"].iloc[0]
                    if not model_filter.empty
                    else "N/A"
                )
                st.session_state["current_oot_target"] = default_oot

    return selected_district, selected_band, selected_model

def render_evaluation_data_source():
    with st.container(border=True):
        st.markdown("##### Evaluation Data Source")
        data_source = st.radio(
            "Select the spatial data extraction paradigm:",
            ["Data Segmen 1 Segmen 2 DAS Ciliwung", "Upload Data"],
            key="eval_data_source",
        )

        if data_source == "Data Segmen 1 Segmen 2 DAS Ciliwung":
            return "internal", None
        else:
            w_size = st.slider(
                "Temporal Window Size (W) for reconstruction",
                min_value=3,
                max_value=7,
                value=5,
                step=2,
                key="eval_custom_w",
            )
            from src.ui.train_model_ui import render_dynamic_uploaders

            render_dynamic_uploaders(window_size=int(w_size), is_disabled=False)
            return "custom", int(w_size)

def render_evaluation_metrics(metrics):
    with st.container(border=True):
        st.markdown("##### 📉 Evaluation Metrics")
        mse_col, rmse_col, r2_col, psnr_col, ssim_col = st.columns(5)
        mse_col.metric("MSE", f"{metrics.get('MSE', 0):.2f}")
        rmse_col.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}")
        r2_col.metric("R² Score", f"{metrics.get('R2', 0):.4f}")
        psnr_col.metric("PSNR", f"{metrics.get('PSNR', 0):.2f} dB")
        ssim_col.metric("SSIM", f"{metrics.get('SSIM', 0):.4f}")


def render_spatial_reconstruction(
    evaluation_title,
    target_date_string,
    image_ground_truth,
    image_reconstructed,
    is_rgb=False,
):
    with st.container(border=True):
        st.markdown(f"##### 🖼️ Spatial Reconstruction Validation: {evaluation_title}")
        st.caption(f"Target Ground Truth Date: **{target_date_string}**")

        cmap = None if is_rgb else "gray"

        image_col1, image_col2 = st.columns(2)
        with image_col1:
            st.markdown("**Ground Truth Observation**")
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            fig1.patch.set_facecolor("none")
            ax1.set_facecolor("none")
            ax1.imshow(image_ground_truth, cmap=cmap)
            ax1.axis("off")
            fig1.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

            buf1 = io.BytesIO()
            fig1.savefig(
                buf1,
                format="png",
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
                facecolor="none",
            )
            st.image(buf1, width="stretch")
            plt.close(fig1)

        with image_col2:
            st.markdown("**Elastic Net Reconstruction**")
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            fig2.patch.set_facecolor("none")
            ax2.set_facecolor("none")
            ax2.imshow(image_reconstructed, cmap=cmap)
            ax2.axis("off")
            fig2.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

            buf2 = io.BytesIO()
            fig2.savefig(
                buf2,
                format="png",
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
                facecolor="none",
            )
            st.image(buf2, width="stretch")
            plt.close(fig2)


def render_evaluation_download(metrics_data, district_name):
    """Renders download button for evaluation summary metrics."""
    with st.container(border=True):
        st.markdown("##### 📥 Export Evaluation Summary")
        df_export = (
            pd.DataFrame(metrics_data)
            if isinstance(metrics_data, list)
            else pd.DataFrame([metrics_data])
        )
        csv_data = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Evaluation Summary (CSV)",
            data=csv_data,
            file_name=f"Evaluation_Summary_{district_name}.csv",
            mime="text/csv",
            type="primary",
            width="stretch",
        )
