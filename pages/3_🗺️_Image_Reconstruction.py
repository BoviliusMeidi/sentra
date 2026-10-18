import os
import sys
import uuid
import streamlit as st
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
from src.config import KECAMATAN_LIST, ALL_BANDS
from src.data_pipeline.temporal_data_discoverer import TemporalDataDiscoverer
from src.controllers.image_reconstruction_orchestrator import (
    ImageReconstructionOrchestrator,
)
from src.ui.image_reconstruction_ui import (
    apply_reconstruction_custom_css,
    render_reconstruction_parameters_card,
    render_reconstruction_results,
    render_download_assets,
    render_custom_reconstruction_uploaders,
    generate_rgb_composite_rgba,
)
from src.utils.temp_workspace_manager import TempWorkspaceManager

st.set_page_config(page_title="Image Reconstruction", page_icon="🖼️", layout="wide")
apply_reconstruction_custom_css()

st.markdown("### 🖼️ Spatial Image Reconstruction")

selected_district, selected_band, selected_model = (
    render_reconstruction_parameters_card(KECAMATAN_LIST, ALL_BANDS)
)

if selected_model:
    with st.container(border=True):
        st.markdown("##### Reconstruction Data Source")
        recon_mode = st.radio(
            "Select the spatial data extraction paradigm:",
            ["Data Segmen 1 Segmen 2 DAS Ciliwung", "Upload Data"],
            key="recon_data_source",
        )

        if recon_mode == "Data Segmen 1 Segmen 2 DAS Ciliwung":
            temporal_data_discoverer = TemporalDataDiscoverer()
            available_dates_catalog = temporal_data_discoverer.process(
                config.MICRO_ROOT, selected_district
            )
            available_years = sorted(
                list(set([d[0] for d in available_dates_catalog])), reverse=True
            )

            db_sub_mode = st.radio(
                "Operation Mode",
                [
                    "Gap-Filling",
                    "Reconstruct from Scratch",
                ],
                key="db_sub_mode",
            )

            if db_sub_mode == "Gap-Filling":
                col_y, col_d = st.columns(2)
                with col_y:
                    sel_year = st.selectbox(
                        "Select Reference Year",
                        available_years if available_years else ["N/A"],
                    )
                with col_d:
                    dates_in_yr = [
                        d[1] for d in available_dates_catalog if d[0] == sel_year
                    ]

                    formatted_date_map = {}
                    for d_str in dates_in_yr:
                        try:
                            dt_obj = datetime.strptime(d_str, "%Y%m%d")
                            formatted_date_map[dt_obj.strftime("%d %B %Y")] = d_str
                        except Exception:
                            formatted_date_map[d_str] = d_str

                    selected_display_date = st.selectbox(
                        "Select Target Date",
                        (
                            list(formatted_date_map.keys())
                            if formatted_date_map
                            else ["N/A"]
                        ),
                    )
                    sel_date = formatted_date_map.get(
                        selected_display_date, selected_display_date
                    )

                recon_action_mode = "gap_fill"
                target_date_str = sel_date
            else:
                custom_date = st.date_input(
                    "Select Virtual Target Date", datetime.today()
                )
                recon_action_mode = "scratch"
                target_date_str = custom_date.strftime("%Y%m%d")

            window_size = 5
        else:
            window_size = 5
            render_custom_reconstruction_uploaders(prefix="recon")

    if st.button("Image Reconstruction", type="primary", use_container_width=True):
        orchestrator = ImageReconstructionOrchestrator()

        if recon_mode == "Data Segmen 1 Segmen 2 DAS Ciliwung":
            bands_to_run = (
                ALL_BANDS if selected_band == "All Bands" else [selected_band]
            )
            tif_bytes_dict = {}
            band_visuals = {}

            with st.spinner(
                f"Executing internal reconstruction for {selected_district}..."
            ):
                success_all = True
                error_msg = ""
                for b in bands_to_run:
                    res = orchestrator.execute_internal_reconstruction(
                        district=selected_district,
                        band=b,
                        model_base_name=selected_model,
                        target_date_str=target_date_str,
                        mode=recon_action_mode,
                        window_size=window_size,
                    )
                    if res["success"]:
                        tif_bytes_dict[b] = res["tif_bytes"]
                        band_visuals[b] = res
                    else:
                        success_all = False
                        error_msg = res["error"]
                        break

                if success_all:
                    st.success("Reconstruction successfully concluded.")
                    if selected_band == "All Bands":
                        tab_labels = ALL_BANDS + ["True Color RGB"]
                        tabs = st.tabs(tab_labels)
                        for idx, b in enumerate(ALL_BANDS):
                            with tabs[idx]:
                                render_reconstruction_results(
                                    visual_results=band_visuals[b],
                                    title_metadata=f"{selected_district} - Band {b}",
                                )
                        with tabs[-1]:
                            rgb_orig = generate_rgb_composite_rgba(
                                band_visuals["B04"]["orig_array"],
                                band_visuals["B03"]["orig_array"],
                                band_visuals["B02"]["orig_array"],
                            )
                            rgb_recon = generate_rgb_composite_rgba(
                                band_visuals["B04"]["patched_array"],
                                band_visuals["B03"]["patched_array"],
                                band_visuals["B02"]["patched_array"],
                            )
                            rgb_visuals = {
                                "spatial_bounds": band_visuals["B02"]["spatial_bounds"],
                                "rgba_original": rgb_orig,
                                "rgba_reconstructed": rgb_recon,
                                "is_synthetic": band_visuals["B02"].get("is_synthetic", False)
                            }
                            render_reconstruction_results(
                                rgb_visuals,
                                title_metadata=f"{selected_district} - True Color RGB",
                                is_rgb=True,
                            )
                    else:
                        render_reconstruction_results(
                            band_visuals[selected_band],
                            title_metadata=f"{selected_district} - {selected_band}",
                        )

                    render_download_assets(
                        tif_bytes_dict, selected_district, target_date_str
                    )
                else:
                    st.error(error_msg)
        else:
            missing_files = False
            for b in ALL_BANDS:
                if not st.session_state.get(f"recon_target_{b}"):
                    missing_files = True
            if not st.session_state.get("recon_shp"):
                missing_files = True

            if missing_files:
                st.error(
                    "Execution Failed: Please upload the required shapefile bundle and all target imagery files (6 Bands)."
                )
            else:
                with st.spinner("Processing custom spatial upload reconstruction..."):
                    session_id = str(uuid.uuid4())
                    temp_mgr = TempWorkspaceManager()
                    try:
                        shp_path = temp_mgr.save_uploaded_file(
                            st.session_state.recon_shp, session_id
                        )
                        target_scl_path = (
                            temp_mgr.save_uploaded_file(
                                st.session_state.get("recon_target_scl"), session_id
                            )
                            if st.session_state.get("recon_target_scl")
                            else None
                        )

                        target_paths = {
                            b: temp_mgr.save_uploaded_file(
                                st.session_state[f"recon_target_{b}"], session_id
                            )
                            for b in ALL_BANDS
                        }

                        bands_to_run = (
                            ALL_BANDS
                            if selected_band == "All Bands"
                            else [selected_band]
                        )
                        tif_bytes_dict = {}
                        band_visuals = {}
                        success_all = True
                        error_msg = ""

                        for b in bands_to_run:
                            res = orchestrator.execute_custom_reconstruction(
                                district=selected_district,
                                band=b,
                                model_base_name=selected_model,
                                target_path=target_paths[b],
                                target_scl_path=target_scl_path,
                                shp_path=shp_path,
                            )
                            if res["success"]:
                                tif_bytes_dict[b] = res["tif_bytes"]
                                band_visuals[b] = res
                            else:
                                success_all = False
                                error_msg = res["error"]
                                break

                        if success_all:
                            st.success("Custom Reconstruction successfully concluded.")
                            if selected_band == "All Bands":
                                tab_labels = ALL_BANDS + ["True Color RGB"]
                                tabs = st.tabs(tab_labels)
                                for idx, b in enumerate(ALL_BANDS):
                                    with tabs[idx]:
                                        render_reconstruction_results(
                                            band_visuals[b],
                                            title_metadata=f"Custom Upload - Band {b}",
                                        )

                                with tabs[-1]:
                                    rgb_orig = generate_rgb_composite_rgba(
                                        band_visuals["B04"]["orig_array"],
                                        band_visuals["B03"]["orig_array"],
                                        band_visuals["B02"]["orig_array"],
                                    )
                                    rgb_recon = generate_rgb_composite_rgba(
                                        band_visuals["B04"]["patched_array"],
                                        band_visuals["B03"]["patched_array"],
                                        band_visuals["B02"]["patched_array"],
                                    )

                                    rgb_visuals = {
                                        "spatial_bounds": band_visuals["B02"]["spatial_bounds"],
                                        "rgba_original": rgb_orig,
                                        "rgba_reconstructed": rgb_recon,
                                        "is_synthetic": band_visuals["B02"].get("is_synthetic", False)
                                    }
                                    
                                    render_reconstruction_results(
                                        rgb_visuals,
                                        title_metadata="Custom Upload - True Color RGB",
                                        is_rgb=True,
                                    )
                            else:
                                render_reconstruction_results(
                                    band_visuals[selected_band],
                                    title_metadata=f"Custom Upload - {selected_band}",
                                )

                            render_download_assets(
                                tif_bytes_dict, selected_district, "CustomOutput"
                            )
                        else:
                            st.error(error_msg)

                    except Exception as e:
                        st.error(f"Error processing custom reconstruction: {str(e)}")
                    finally:
                        temp_mgr.cleanup_session(session_id)