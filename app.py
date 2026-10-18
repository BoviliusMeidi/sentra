import streamlit as st
import os
from src.utils.ui_helpers import apply_section_backgrounds, get_image_base64, render_custom_sidebar

logo_path = "assets/images/logo.png"
st.set_page_config(
    page_title="SENTRA | Image Reconstruction",
    page_icon=logo_path,
    layout="wide",
)

def render_landing_page():
    css_path = "assets/style.css"
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.html(f"<style>{f.read()}</style>")

    section_backgrounds = [
        {"selector": "#logo", "image": "assets/images/logo.png"},
        {"selector": "#section-1", "image": "assets/images/hero_bg.webp"},
        {"selector": "#sec2", "image": "assets/images/about_bg.webp"},
        {"selector": "#sec3", "image": "assets/images/core_features_bg.webp"},
        {"selector": "#sec4", "image": "assets/images/hero_bg.webp"},
        {"selector": "#sec5", "image": "assets/images/help_bg.webp"},
        {"selector": "#sec6", "image": "assets/images/footer_bg.webp"},
    ]
    apply_section_backgrounds(section_backgrounds)

    html_path = "assets/home.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        bovilius_b64 = get_image_base64("assets/images/behind_the_screen/bovilius_meidi.webp")
        dyah_b64 = get_image_base64("assets/images/behind_the_screen/dyah_erny_herwindiati.webp")
        janson_b64 = get_image_base64("assets/images/behind_the_screen/janson_hendryli.webp")
        logo_b64 = get_image_base64("assets/images/logo.png")

        html_content = html_content.replace('id="img-bovilius"', f'src="{bovilius_b64}"')
        html_content = html_content.replace('id="img-dyah"', f'src="{dyah_b64}"')
        html_content = html_content.replace('id="img-janson"', f'src="{janson_b64}"')
        html_content = html_content.replace('id="footer-logo"', f'src="{logo_b64}"')

        st.html(html_content)

page_home  = st.Page(render_landing_page, title="Home", icon="🏠", default=True)
page_train = st.Page("pages/1_⚙️_Train_Model.py", title="Train Model", icon="⚙️")
page_eval  = st.Page("pages/2_📊_Evaluate_Model.py", title="Evaluate Model", icon="📊")
page_recon = st.Page("pages/3_🗺️_Image_Reconstruction.py", title="Image Reconstruction", icon="🗺️")
page_help  = st.Page("pages/4_❓_Help.py", title="Help", icon="❓")

pg = st.navigation([page_home, page_train, page_eval, page_recon, page_help], position="hidden")
if pg.title != "Home":
    render_custom_sidebar()
pg.run()