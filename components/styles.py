import streamlit as st
from pathlib import Path


def load_css() -> None:
    """
    Load the application's external CSS file.
    """
    css_path = Path(__file__).parent.parent / "assets" / "style.css"

    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True,
            )