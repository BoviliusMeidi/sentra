import pandas as pd
import streamlit as st
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ui.train_model_ui import (
    apply_training_custom_css,
    render_internal_scope,
    render_hyperparameters,
    render_dynamic_uploaders,
)
from src.controllers.train_model_orchestrator import TrainModelOrchestrator
from src.config import ALL_BANDS

apply_training_custom_css()
st.markdown("### ⚙️ Train Model")

def extract_configs_from_state(prefix: str):
    return {
        "w": st.session_state[f"{prefix}_w"],
        "seed": st.session_state[f"{prefix}_seed"],
        "alphas": st.session_state[f"{prefix}_alphas"],
        "l1s": st.session_state[f"{prefix}_l1s"],
        "samples_list": st.session_state[f"{prefix}_samples"],
        "ratios_list": st.session_state[f"{prefix}_ratios"],
    }


if "is_training" not in st.session_state:
    st.session_state.is_training = False

is_disabled = st.session_state.is_training

tab1, tab2 = st.tabs(
    ["Ciliwung (Segmen 1 & Segmen 2)", "Upload Data"]
)

with tab1:
    render_internal_scope(is_disabled=is_disabled)
    st.markdown("---")
    render_hyperparameters(is_disabled=is_disabled, prefix="int")

    if not is_disabled:
        if st.button("Training Model", type="primary", width="stretch"):
            if not st.session_state.get("int_model_name", "").strip():
                st.error("Please provide a valid Model Name before executing.")
            else:
                st.session_state.active_mode = "internal"
                st.session_state.is_training = True
                st.session_state.pop("summary_df", None)
                st.rerun()

with tab2:
    st.text_input(
        "Base Model Name",
        placeholder="e.g., custom_area",
        key="cust_model_name",
        disabled=is_disabled,
    )
    st.markdown("---")
    render_hyperparameters(is_disabled=is_disabled, prefix="cust")
    st.markdown("---")

    current_w = st.session_state.get("cust_w", 5)
    render_dynamic_uploaders(window_size=current_w, is_disabled=is_disabled)

    if not is_disabled:
        if st.button("Training Models", type="primary", width="stretch"):
            if not st.session_state.cust_model_name.strip():
                st.error("Please provide a valid Base Model Name.")
            else:
                missing = False
                for i in range(1, current_w + 1):
                    for b in ALL_BANDS:
                        if not st.session_state.get(f"up_hist_{i}_{b}"):
                            missing = True
                for b in ALL_BANDS:
                    if not st.session_state.get(f"up_gt_{b}"):
                        missing = True

                if missing:
                    st.error(
                        f"Please ensure all {current_w * 6} historical files and 6 Ground Truth files are uploaded."
                    )
                else:
                    st.session_state.active_mode = "custom"
                    st.session_state.is_training = True
                    st.session_state.pop("summary_df", None)
                    st.rerun()

if is_disabled:
    st.markdown("---")
    st.markdown("##### Logs Training")

    if st.button("Cancel / Reset", width="stretch"):
        st.session_state.is_training = False
        st.session_state.pop("summary_df", None)
        st.rerun()

    if "summary_df" not in st.session_state:
        progress_bar = st.progress(0)
        start_time = datetime.now()

        with st.status(
            "Executing training workflow...", expanded=True
        ) as status:
            status_container = st.container()
            orchestrator = TrainModelOrchestrator()

            try:
                if st.session_state.active_mode == "internal":
                    configs = extract_configs_from_state("int")
                    st.session_state.summary_df = (
                        orchestrator.execute_internal_pipeline(
                            selected_kec=st.session_state.input_kec,
                            selected_band=st.session_state.input_band,
                            base_name=st.session_state.int_model_name.strip(),
                            config_dict=configs,
                            status_container=status_container,
                            progress_bar=progress_bar,
                        )
                    )
                else:
                    configs = extract_configs_from_state("cust")
                    base_name = st.session_state.get(
                        "cust_model_name", "CustomModel"
                    ).strip()
                    st.session_state.summary_df = orchestrator.execute_custom_pipeline(
                        base_name=base_name,
                        config_dict=configs,
                        session_state_ref=st.session_state,
                        status_container=status_container,
                        progress_bar=progress_bar,
                    )
            except Exception as e:
                st.error(str(e))
                st.session_state.summary_df = pd.DataFrame()

        end_time = datetime.now()
        duration = end_time - start_time
        status.update(
            label=f"Execution completed! Duration: {duration}",
            state="complete",
            expanded=False,
        )

    summary_df = st.session_state.get("summary_df")

    if summary_df is not None and not summary_df.empty:
        st.markdown("###### 📋 Evaluation Summary")
        st.dataframe(summary_df, width="stretch", hide_index=True)

        csv_data = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Summary as CSV",
            data=csv_data,
            file_name=f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            width="stretch",
        )
        st.success("Training process successfully.")
