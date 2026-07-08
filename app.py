import streamlit as st

from config import FEATURES

from components.history import save_prediction, render_history
from components.styles import load_css
from components.sidebar import render_sidebar
from components.header import render_header
from components.forms import (
    store_form,
    promotion_form,
    calendar_form,
    competition_form,
)
from components.result import display_result
from components.footer import render_footer

from utils.loader import load_model, load_model_meta
from utils.validation import validate_inputs
from utils.prediction import build_input_dataframe, run_prediction
 
# ==========================================
# PAGE CONFIG
# ==========================================
 
st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="🏪",
    layout="wide"
)
 
# ==========================================
# CUSTOM CSS
# ==========================================
 
load_css() 
# ==========================================
# MODEL & METADATA LOADING
# ==========================================
 
model = load_model()
meta = load_model_meta()
 
# ==========================================
# SIDEBAR
# ==========================================
 
render_sidebar(meta)
 
# ==========================================
# HEADER
# ==========================================
 
left = render_header(meta, FEATURES)
 
# ==========================================
# INPUT FORMS
# ==========================================
 
with left:
    store_inputs = store_form()
    promotion_inputs = promotion_form()
    calendar_inputs = calendar_form()
    competition_inputs = competition_form()
 
inputs = {
    **store_inputs,
    **promotion_inputs,
    **calendar_inputs,
    **competition_inputs,
}
 
# ==========================================
# PREDICT BUTTON
# ==========================================
 
st.divider()
 
col1, col2, col3 = st.columns([1, 2, 1])
 
with col2:
    predict = st.button(
        "🚀 Predict Sales",
        use_container_width=True
    )
 
# ==========================================
# PREDICTION WORKFLOW
# ==========================================
 
if predict:
 
    # --- 1. Validate inputs ---
    validation_warnings = validate_inputs(
        store_target_enc=inputs["Store_TargetEnc"],
        competition_distance=inputs["CompetitionDistance"],
        competition_distance_missing=inputs["CompetitionDistanceMissing"],
        competition_open_months=inputs["CompetitionOpenMonths"],
        competition_open_missing=inputs["CompetitionOpenMissing"],
        promo2_active_weeks=inputs["Promo2ActiveWeeks"],
        week=int(inputs["Week"]),
        year=int(inputs["Year"]),
    )
 
    if validation_warnings:
        for warning_msg in validation_warnings:
            st.warning(f"⚠️ {warning_msg}")
        st.info(
            "The prediction will still run, but results may be less reliable "
            "for inputs outside the training distribution."
        )

    # --- 2. Build input DataFrame in training column order ---
    input_data = build_input_dataframe(inputs)

    # --- 3. Progress feedback & inference ---
    progress = st.progress(0, text="Preparing data…")
    progress.progress(40, text="Running model…")

    try:
        prediction = run_prediction(model, input_data)
    except Exception as exc:
        st.error(f"❌ Prediction failed: {exc}")
        st.stop()

    progress.progress(100, text="Prediction complete ✅")
    progress.empty()
    save_prediction(inputs, prediction)

    # --- 4. Display results ---
    display_result(prediction, input_data, meta, FEATURES)
    render_history()
# ==========================================
# FOOTER
# ==========================================
 
render_footer(meta)