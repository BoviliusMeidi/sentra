import os
import base64
import streamlit as st

def get_image_base64(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
            ext = image_path.split('.')[-1].lower()
            img_ext = "webp" if ext == "webp" else "png"
            return f"data:image/{img_ext};base64,{encoded}"
    return ""

def apply_section_backgrounds(background_configs):
    css_rules = []

    for config in background_configs:
        selector = config.get("selector")
        img_path = config.get("image")
        overlay = config.get("overlay", "")

        if os.path.exists(img_path):
            with open(img_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()
                img_ext = "webp" if img_path.lower().endswith(".webp") else "png"
                bg_property = f"{overlay}, " if overlay else ""
                bg_property += f"url('data:image/{img_ext};base64,{img_base64}')"

                rule = f"""
                {selector} {{
                    background: {bg_property} center center no-repeat !important;
                    background-size: cover !important;
                }}
                """
                css_rules.append(rule)

    if css_rules:
        st.html(f"<style>{''.join(css_rules)}</style>")

def render_custom_sidebar():
    logo_b64 = get_image_base64("assets/images/logo.png")

    sidebar_html = f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            margin-left: 260px !important;
            width: calc(100vw - 260px) !important;
        }}

        /* SINKRONISASI PADDING ATAS AGAR SEJAJAR DENGAN SIDEBAR */
        [data-testid="stMainBlockContainer"], .block-container {{
            padding-top: 2.5rem !important;
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
        }}

        .scroll-wrapper {{
            left: 260px !important;
            width: calc(100vw - 260px) !important;
        }}

        .navbar {{
            display: none !important;
        }}

        .custom-sidebar {{
            position: fixed;
            top: 0;
            left: 0;
            width: 260px;
            height: 100vh;
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-right: 1px solid #cbd5e1;
            z-index: 999999;
            padding: 2.5rem 1.5rem; /* Padding atas sidebar disamakan dengan padding konten */
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            box-shadow: 4px 0 25px rgba(15, 23, 42, 0.08);
        }}

        .sidebar-brand {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 2.2rem;
            padding-left: 0.5rem;
        }}

        .sidebar-logo {{
            width: 48px;
            height: 48px;
            object-fit: contain;
            filter: drop-shadow(0 2px 8px rgba(2, 132, 199, 0.25));
        }}

        .sidebar-brand-text {{
            font-size: 1.8rem;
            font-weight: 900;
            letter-spacing: 1px;
            background: linear-gradient(135deg, #0f172a 30%, #0284c7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1;
        }}

        .sidebar-nav {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .sidebar-link {{
            text-decoration: none;
            color: #475569;
            font-size: 1.05rem;
            font-weight: 600;
            padding: 0.85rem 1.2rem;
            border-radius: 10px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            background: transparent;
            border: 1px solid transparent;
            display: flex;
            align-items: center;
        }}

        .sidebar-link:hover {{
            color: #0f172a;
            background: rgba(2, 132, 199, 0.1);
            border-color: rgba(2, 132, 199, 0.25);
            transform: translateX(6px);
            box-shadow: 0 4px 15px rgba(2, 132, 199, 0.12);
        }}
    </style>

    <aside class="custom-sidebar" aria-label="Global Sidebar Navigation">
        <div class="sidebar-brand">
            <img src="{logo_b64}" alt="SENTRA Logo" class="sidebar-logo">
            <div class="sidebar-brand-text">SENTRA</div>
        </div>
        <nav class="sidebar-nav">
            <a href="/" target="_self" class="sidebar-link">🏠 Home</a>
            <a href="Train_Model" target="_self" class="sidebar-link">⚙️ Train Model</a>
            <a href="Evaluate_Model" target="_self" class="sidebar-link">📊 Evaluate Model</a>
            <a href="Image_Reconstruction" target="_self" class="sidebar-link">🗺️ Reconstruction</a>
            <a href="Help" target="_self" class="sidebar-link">❓ Help</a>
        </nav>
    </aside>
    """
    st.html(sidebar_html)