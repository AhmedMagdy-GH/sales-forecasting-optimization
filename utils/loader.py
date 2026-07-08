import os
import json
import joblib
import streamlit as st


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """
    Load the serialized LightGBM pipeline.
    """
    model_path = "model.pkl"

    if not os.path.exists(model_path):
        st.error("❌ model.pkl not found.")
        st.stop()

    try:
        return joblib.load(model_path)
    except Exception as exc:
        st.error(f"❌ Failed to load model: {exc}")
        st.stop()


@st.cache_data(show_spinner=False)
def load_model_meta():
    """
    Load metadata for the model.
    """

    fallback = {
        "model": "LightGBM_tuned_v2",
        "r2": 0.8266,
        "mae_orig": 825.52,
        "rmse_orig": 1182.59,
    }

    if not os.path.exists("model_meta.json"):
        return fallback

    try:
        with open("model_meta.json", "r") as f:
            data = json.load(f)

        return {**fallback, **data}

    except Exception:
        return fallback