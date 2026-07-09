


import streamlit as st
import joblib
import json
from pathlib import Path

from config import MODEL_PATH, META_PATH, FEATURES


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """
    Load the serialized XGBoost prediction pipeline from model.pkl.
    Uses @st.cache_resource so the file is read from disk exactly once
    per Streamlit server session, regardless of how many times the
    script re-runs due to user interaction.
    """
    model_path = MODEL_PATH
    if not Path(model_path).exists():
        st.error(
            f"❌ **model.pkl not found.**  "
            f"Please ensure `model.pkl` exists at `{MODEL_PATH}`."
        )
        st.stop()
    try:
        return joblib.load(model_path)
    except Exception as exc:
        st.error(f"❌ **Failed to load model:** {exc}")
        st.stop()


@st.cache_data(show_spinner=False)
def load_model_meta():
    """
    Load optional model_meta.json if it exists alongside the app.
    Returns a dict with known-good fallback values when the file is absent,
    so the UI always has something to display.
    """
    meta_path = META_PATH
    fallback = {
        "model":      "XGBoost Regressor",
        "r2":         0.8306,
        "mae_orig":   815.18,
        "rmse_orig":  1150.54,
        "mape":       0.1370,
        "features":   FEATURES,
    }
    if not Path(meta_path).exists():
        return fallback
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Merge with fallback so missing keys don't crash the UI
        return {**fallback, **data}
    except Exception:
        return fallback
