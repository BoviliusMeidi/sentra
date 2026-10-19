import sys
import os
import uuid
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import KECAMATAN_LIST, ALL_BANDS
from src.ui.evaluate_model_ui import (
    apply_evaluate_custom_css,
    render_target_parameters_card,
    render_custom_evaluation_uploaders,
    render_evaluation_metrics,
    render_spatial_reconstruction,
    render_evaluation_download
)
from src.controllers.evaluate_model_orchestrator import EvaluateModelOrchestrator
from src.utils.temp_workspace_manager import TempWorkspaceManager

st.set_page_config(page_title="Evaluate Model", page_icon="📊", layout="wide")
apply_evaluate_custom_css()
st.markdown("### 📊 Evaluate Model")

selected_district, selected_band, selected_model = render_target_parameters_card(
    KECAMATAN_LIST, ALL_BANDS
)

if selected_model:
    if selected_district == "Custom Upload":
        data_source = "custom"
        render_custom_evaluation_uploaders()
    else:
        data_source = "internal"

    current_pipeline_identifier = f"{selected_district}_{selected_band}_{selected_model}_{data_source}"
    if st.session_state.get("active_eval_identifier") != current_pipeline_identifier:
        st.session_state["is_eval_pipeline_completed"] = False
        st.session_state["active_eval_identifier"] = current_pipeline_identifier

    if not st.session_state.get("is_eval_pipeline_completed", False):
        if st.button("Evaluate Model", type="primary", use_container_width=True):
            st.session_state["trigger_eval_execution"] = True

    if st.session_state.get("trigger_eval_execution", False):
        orchestrator = EvaluateModelOrchestrator()

        if data_source == "internal":
            oot_date = st.session_state.get("current_oot_target", "Auto-Detect")

            with st.spinner(f"Evaluating Data for {selected_district}..."):
                if selected_band == "All Bands":
                    res = orchestrator.execute_all_bands_evaluation(
                        district=selected_district,
                        model_base_name=selected_model,
                        window_size=5,
                        target_date_str=oot_date
                    )
                else:
                    res = orchestrator.execute_single_band_evaluation(
                        district=selected_district,
                        band=selected_band,
                        model_base_name=selected_model,
                        window_size=5,
                        target_date_str=oot_date
                    )

                if res["success"]:
                    st.session_state["eval_payload"] = res
                    st.session_state["is_eval_pipeline_completed"] = True
                    st.session_state["trigger_eval_execution"] = False
                    st.rerun()
                else:
                    st.error(res["error"])
                    st.session_state["trigger_eval_execution"] = False

        else:
            missing_files = False
            bands_to_check = ALL_BANDS if selected_band == "All Bands" else [selected_band]

            for b in bands_to_check:
                if not st.session_state.get(f"eval_gt_{b}"):
                    missing_files = True

            if missing_files:
                st.error("Execution Failed: Please ensure Ground Truth target files are fully uploaded.")
                st.session_state["trigger_eval_execution"] = False
            else:
                with st.spinner("Processing custom spatial data evaluation..."):
                    session_id = str(uuid.uuid4())
                    temp_mgr = TempWorkspaceManager()

                    try:
                        shp_file = st.session_state.get("eval_shp")
                        shp_path = temp_mgr.save_uploaded_file(shp_file, session_id) if shp_file else None

                        gt_scl_file = st.session_state.get("eval_gt_scl")
                        gt_scl_path = temp_mgr.save_uploaded_file(gt_scl_file, session_id) if gt_scl_file else None

                        if selected_band == "All Bands":
                            gt_paths = {b: temp_mgr.save_uploaded_file(st.session_state[f"eval_gt_{b}"], session_id) for b in ALL_BANDS}
                            res = orchestrator.execute_custom_all_bands_evaluation(
                                district=selected_district,
                                model_base_name=selected_model,
                                gt_paths=gt_paths,
                                gt_scl_path=gt_scl_path,
                                shp_path=shp_path,
                            )
                        else:
                            gt_path = temp_mgr.save_uploaded_file(st.session_state[f"eval_gt_{selected_band}"], session_id)
                            res = orchestrator.execute_custom_single_band_evaluation(
                                district=selected_district,
                                band=selected_band,
                                model_base_name=selected_model,
                                gt_path=gt_path,
                                gt_scl_path=gt_scl_path,
                                shp_path=shp_path,
                            )

                        if res["success"]:
                            st.session_state["eval_payload"] = res
                            st.session_state["is_eval_pipeline_completed"] = True
                            st.session_state["trigger_eval_execution"] = False
                            st.rerun()
                        else:
                            st.error(res["error"])
                            st.session_state["trigger_eval_execution"] = False

                    except Exception as e:
                        st.error(f"An unexpected error occurred during custom file processing: {str(e)}")
                        st.session_state["trigger_eval_execution"] = False
                    finally:
                        temp_mgr.cleanup_session(session_id)

    if st.session_state.get("is_eval_pipeline_completed", False):
        res = st.session_state["eval_payload"]

        st.success("Evaluation pipeline successfully concluded.")
        st.markdown("#### Spatial Comparison")

        if selected_band == "All Bands":
            tab_labels = ALL_BANDS + ["True Color RGB"]
            tabs = st.tabs(tab_labels)

            for idx, band_name in enumerate(ALL_BANDS):
                with tabs[idx]:
                    band_data = res["band_raw_results"].get(band_name)
                    if band_data and band_data["success"]:
                        render_evaluation_metrics(band_data["metrics"])
                        render_spatial_reconstruction(
                            evaluation_title=f"{'Custom Upload' if data_source == 'custom' else selected_district} - Spectral Band {band_name}",
                            target_date_string=res["target_date_string"] if data_source == "internal" else "Custom Output",
                            image_ground_truth=band_data["image_ground_truth"],
                            image_reconstructed=band_data["image_reconstructed"],
                            is_rgb=False,
                        )
                    else:
                        st.warning(f"Spatial comparison data for band {band_name} is unavailable.")

            with tabs[-1]:
                if res.get("rgb_ground_truth") is not None and res.get("rgb_reconstructed") is not None:
                    render_spatial_reconstruction(
                        evaluation_title=f"{'Custom Upload' if data_source == 'custom' else selected_district} - True Color RGB Composite",
                        target_date_string=res["target_date_string"] if data_source == "internal" else "Custom Output",
                        image_ground_truth=res["rgb_ground_truth"],
                        image_reconstructed=res["rgb_reconstructed"],
                        is_rgb=True,
                    )
                else:
                    st.warning("Incomplete spectral bands (B04, B03, B02); unable to synthesize True Color RGB composite.")

            download_label = f"{selected_district}_CustomUpload" if data_source == "custom" else selected_district
            render_evaluation_download(res["metrics_list"], download_label)

        else:
            render_evaluation_metrics(res["metrics"])
            render_spatial_reconstruction(
                evaluation_title=f"{'Custom Upload' if data_source == 'custom' else selected_district} - {selected_band}",
                target_date_string=res["target_date_string"] if data_source == "internal" else "Custom Output",
                image_ground_truth=res["image_ground_truth"],
                image_reconstructed=res["image_reconstructed"],
                is_rgb=False,
            )
            download_label = f"{selected_district}_CustomUpload_{selected_band}" if data_source == "custom" else f"{selected_district}_{selected_band}"
            render_evaluation_download(res["metrics"], download_label)

        if st.button("Reset Evaluation", use_container_width=True):
            st.session_state["is_eval_pipeline_completed"] = False
            st.session_state.pop("eval_payload", None)
            st.rerun()