import streamlit as st

st.markdown(
    """
    <style>
        .block-container { padding-top: 2.5rem !important; padding-left: 2.5rem !important; padding-right: 2.5rem !important; }
        .help-subheader { color: #0284c7; font-weight: 600; margin-bottom: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("### ❓ Help")

train_model_tab, evaluate_model_tab, image_reconstruction_tab = st.tabs(
    ["⚙️ Train Model", "📊 Evaluate Model", "🖼️ Image Reconstruction"]
)

with train_model_tab:
    with st.container(border=True):
        st.markdown("<div class='help-subheader'>Overview</div>", unsafe_allow_html=True)
        st.markdown("""
        The **Train Model** module serves as the primary optimization engine of **SENTRA**, constructed to learn surface reflectance patterns using **Elastic Net Regression**. It processes multi-spectral Sentinel-2 datasets—both pre-loaded regional data and custom datasets uploaded via the **User Upload** feature—through column-wise *Z-Score standardization* (StandardScaler) and optimizes weight coefficients via *Coordinate Descent*.
        """)

    with st.container(border=True):
        st.markdown("<div class='help-subheader'>Configuration & Execution Guide</div>", unsafe_allow_html=True)

        left_column, right_column = st.columns(2)

        with left_column:
            st.markdown("**1. Data Source & Scope**")
            st.markdown("""
            - **Regional Preset**: Train models using built-in watershed datasets.
            - **User Upload**: Upload custom multi-band GeoTIFF arrays or tabular datasets directly through the interface to train models on custom spatial regions.
            - **Processing Scope**: Choose between *Single Model* (specific band/district), *Regional Batch*, or *Full Batch Execution*.
            """)

            st.markdown("**2. Temporal Window (W)**")
            st.markdown("""
            Defines the depth of historical sequences. For instance, setting `W=5` extracts feature matrices from the 5 preceding cloud-free observation dates prior to the target timestamp.
            """)

        with right_column:
            st.markdown("**3. Regularization Space**")
            st.markdown("""
            - **Alpha ($\\lambda$)**: Controls the overall penalty strength applied to the combined $L_1$ and $L_2$ loss function.
            - **L1 Ratio ($\\alpha$)**: Manages the balance between Lasso and Ridge penalties; `1.0` represents pure Lasso for sparse feature selection, while lower values incorporate Ridge to stabilize multicollinearity.
            """)

            st.markdown("**4. Data Sampling & Validation**")
            st.markdown("""
            Manages spatial pixel sampling limits to ensure memory efficiency during high-dimensional matrix operations, alongside configurable train-test split allocations.
            """)

with evaluate_model_tab:
    with st.container(border=True):
        st.markdown("<div class='help-subheader'>Overview</div>", unsafe_allow_html=True)
        st.markdown("""
        The **Evaluate Model** module assesses the mathematical robustness and pure-pixel accuracy of the trained Elastic Net models, providing rigorous statistical validation for thesis documentation and custom user uploads.
        """)

    with st.container(border=True):
        st.markdown("<div class='help-subheader'>Evaluation Modes</div>", unsafe_allow_html=True)

        st.markdown("**🔍 Evaluation & Visualization**")
        st.markdown("""
        Designed for in-depth inspection of specific model performance.
        - Computes pure mathematical metrics strictly on cloud-free test pixels within the temporal window (`MSE, RMSE, R², PSNR, SSIM`).
        - Renders spatial comparative visuals between the *Ground Truth* and the *Reconstructed Gap-Filled* output.
        - Supports custom evaluation on user-uploaded imagery datasets.
        """)

with image_reconstruction_tab:
    with st.container(border=True):
        st.markdown("<div class='help-subheader'>Overview</div>", unsafe_allow_html=True)
        st.markdown("""
        The **Image Reconstruction** module is the operational core of the system. It applies optimized regression matrices to logically impute cloud-obscured pixels (Gap-Filling) while strictly preserving true spectral fidelity without generative hallucination. Enhanced with **User Upload** capabilities to process external cloudy imagery.
        """)

    with st.container(border=True):
        st.markdown("<div class='help-subheader'>Key Novelties & Features</div>", unsafe_allow_html=True)

        left_column, right_column = st.columns(2)

        with left_column:
            st.markdown("**1. Operation Modes & Upload**")
            st.markdown("""
            - **Targeted Gap-Filling**: Utilizes existing cloudy imagery metadata (either preset or user-uploaded) to predict and replace missing ($NaN$) values within cloud-masked regions.
            - **Synthetic Generation**: Evaluates custom virtual target dates, synthesizing terrain reflectance patterns purely based on learned historical temporal trends.
            """)

        with right_column:
            st.markdown("**2. Asset Export**")
            st.markdown("""
            Exports reconstructed spatial layers into standard **GeoTIFF (.tif)** formats, packaged neatly into compressed archives for seamless integration into external GIS software.
            """)
