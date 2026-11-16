from pathlib import Path

import streamlit as st

from config import APP_NAME, APP_SUBTITLE, BASE_DIR
from modules.database import initialize_database
from pages import analytics, dashboard, history, image_upload, results

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_database()
style_path = BASE_DIR / "assets" / "style.css"
try:
    style = style_path.read_text(encoding="utf-8")
except OSError as exc:
    st.error(f"The application stylesheet could not be loaded: {exc}")
    style = ""
st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        '<div style="font-size:2rem">🅿️</div>'
        '<div style="font-family:Space Grotesk,sans-serif;font-size:1.15rem;'
        'font-weight:700;color:#eaf6ff;letter-spacing:.02em">SMART PARKING <span '
        'style="color:#22d3ee">AI</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(APP_SUBTITLE)
    st.markdown("---")
    selected_page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Upload Parking Image",
            "Detection Results",
            "Parking Analytics",
            "Detection History",
            "About Project",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("IMAGE-ONLY ANALYSIS")
    st.caption("YOLOv8 Nano · OpenCV · SQLite")

page_renderers = {
    "Dashboard": dashboard.render,
    "Upload Parking Image": image_upload.render,
    "Detection Results": results.render,
    "Parking Analytics": analytics.render,
    "Detection History": history.render,
}

def render_about() -> None:
    st.markdown('<div class="eyebrow">ABOUT THE PLATFORM</div>', unsafe_allow_html=True)
    st.title("Smart Parking AI")
    st.markdown(
        """
        **Smart Parking AI** analyzes uploaded parking-lot photographs to classify
        marked parking spaces as occupied or empty. Vehicle boxes are detected with
        the pretrained YOLOv8 Nano model, and each vehicle is associated with a
        parking-space region using a configurable overlap threshold.

        ### Parking-space regions
        Automatic image-processing proposals are offered when visible outlines can
        be found. Because parking layouts differ, proposals are shown for review.
        For reliable counts, draw a rectangle around each stall. Save that layout
        under a viewpoint name to reuse it for future images from the same view.

        ### Color legend
        - 🟩 **Green:** empty parking space
        - 🟥 **Red:** occupied parking space
        - 🟦 **Blue:** detected vehicle

        ### Privacy and input scope
        The application accepts still JPG/JPEG/PNG uploads only. It has no camera,
        video, or streaming features. Detection records and annotated images are
        stored locally in the project directory.

        The vehicle model is pretrained; this project does not claim custom
        training or a particular detection accuracy. Results depend on image
        quality and the accuracy of the parking-space regions.
        """
    )

if selected_page == "About Project":
    render_about()
else:
    page_renderers[selected_page]()
