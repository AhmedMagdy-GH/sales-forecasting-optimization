import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

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

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
}

.card {
    background: #1c1f26;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #2f3542;
    box-shadow: 0px 0px 12px rgba(0, 0, 0, .35);
}

.metric-card {
    background: #262730;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 15px;
}

.metric-title {
    color: #8f9bb3;
    font-size: 18px;
}

.metric-value {
    color: #00d26a;
    font-size: 32px;
    font-weight: bold;
}

.result-card {
    background: #133c27;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
}

.result-title {
    font-size: 22px;
    color: white;
}

.result-value {
    font-size: 46px;
    color: #00ff84;
    font-weight: bold;
}

.validation-warning {
    background: #2d2000;
    border-left: 4px solid #ffc107;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 6px 0;
    color: #ffc107;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS — Feature list in exact training order
# ==========================================

FEATURES = [
    'Store_TargetEnc',
    'DayOfWeek',
    'CompetitionDistance',
    'CompetitionDistanceMissing',
    'CompetitionOpenMissing',
    'StateHoliday',
    'SchoolHoliday',
    'Promo',
    'Promo2',
    'StoreType',
    'Assortment',
    'Year',
    'Week',
    'IsWeekend',
    'IsMonthStart',
    'IsMonthEnd',
    'CompetitionOpenMonths',
    'Promo2ActiveWeeks',
    'IsPromo2Active'
]

# Training distribution reference ranges for validation
FEATURE_RANGES = {
    "Store_TargetEnc":      (5.0, 12.0),
    "CompetitionDistance":  (0.0, 100_000.0),
    "CompetitionOpenMonths": (0.0, 600.0),
    "Promo2ActiveWeeks":    (0.0, 260.0),
    "Week":                 (1, 53),
    "Year":                 (2013, 2030),
}

# ==========================================
# MODEL LOADING — cached so it runs only once per session
# ==========================================

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """
    Load the serialized LightGBM pipeline from model.pkl.
    Uses @st.cache_resource so the file is read from disk exactly once
    per Streamlit server session, regardless of how many times the
    script re-runs due to user interaction.
    """
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        st.error(
            "❌ **model.pkl not found.**  "
            "Please ensure `model.pkl` is in the same directory as `app.py`."
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
    meta_path = "model_meta.json"
    fallback = {
        "model":      "LightGBM_tuned_v2",
        "r2":         0.8266,
        "mae_orig":   825.52,
        "rmse_orig":  1182.59,
        "features":   FEATURES,
    }
    if not os.path.exists(meta_path):
        return fallback
    try:
        with open(meta_path, "r") as fh:
            data = json.load(fh)
        # Merge with fallback so missing keys don't crash the UI
        return {**fallback, **data}
    except Exception:
        return fallback


model = load_model()
meta  = load_model_meta()

# ==========================================
# INPUT VALIDATION
# ==========================================

def validate_inputs(
    store_target_enc: float,
    competition_distance: float,
    competition_open_months: float,
    promo2_active_weeks: float,
    week: int,
    year: int,
) -> list[str]:
    """
    Return a list of human-readable warning strings for inputs that fall
    outside the ranges seen during training.  An empty list means all
    inputs are within expected bounds.
    """
    warnings_out = []

    lo, hi = FEATURE_RANGES["Store_TargetEnc"]
    if not (lo <= store_target_enc <= hi):
        warnings_out.append(
            f"Store Target Encoding ({store_target_enc:.4f}) is outside the "
            f"training range ({lo}–{hi}).  Predictions may be unreliable."
        )

    lo, hi = FEATURE_RANGES["CompetitionDistance"]
    if not (lo <= competition_distance <= hi):
        warnings_out.append(
            f"Competition Distance ({competition_distance:,.0f} m) is outside "
            f"the expected range ({lo:,.0f}–{hi:,.0f} m)."
        )

    if competition_distance < 0:
        warnings_out.append("Competition Distance cannot be negative.")

    lo, hi = FEATURE_RANGES["CompetitionOpenMonths"]
    if not (lo <= competition_open_months <= hi):
        warnings_out.append(
            f"Competition Open Months ({competition_open_months}) is unusually "
            f"large (expected ≤ {hi}).  Please verify the value."
        )

    lo, hi = FEATURE_RANGES["Promo2ActiveWeeks"]
    if not (lo <= promo2_active_weeks <= hi):
        warnings_out.append(
            f"Promo2 Active Weeks ({promo2_active_weeks}) is outside the "
            f"expected range ({lo}–{hi})."
        )

    lo, hi = FEATURE_RANGES["Week"]
    if not (lo <= week <= hi):
        warnings_out.append(
            f"Week ({week}) must be between {lo} and {hi}."
        )

    lo, hi = FEATURE_RANGES["Year"]
    if not (lo <= year <= hi):
        warnings_out.append(
            f"Year ({year}) must be between {lo} and {hi}."
        )

    return warnings_out


# ==========================================
# PREDICTION HELPERS
# ==========================================

def build_input_dataframe(inputs: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame from the user's inputs, with columns in
    the exact order the model was trained on.  Column ordering is enforced
    explicitly to guard against future dict-ordering surprises.
    """
    return pd.DataFrame([inputs])[FEATURES]


def run_prediction(input_df: pd.DataFrame) -> float:
    """
    Run inference through the cached LightGBM pipeline and invert the
    log-transform applied during training (np.expm1 = inverse of np.log1p).
    Returns the predicted daily sales in Euros.
    """
    log_pred = model.predict(input_df)[0]
    return float(np.expm1(log_pred))


# ==========================================
# RESULT DISPLAY
# ==========================================

def get_sales_status(prediction: float) -> tuple[str, str]:
    """Return (status_label, hex_color) based on the predicted sales value."""
    if prediction < 3_000:
        return "🔴 Low Sales Expected",      "#dc3545"
    elif prediction < 7_000:
        return "🟡 Moderate Sales Expected", "#ffc107"
    else:
        return "🟢 High Sales Expected",     "#28a745"


def display_result(prediction: float, input_df: pd.DataFrame) -> None:
    """Render the full results section: input summary, prediction card, metrics."""
    sales_status, status_color = get_sales_status(prediction)

    st.divider()
    st.subheader("📋 Input Summary")
    st.dataframe(input_df, use_container_width=True)

    st.divider()
    st.markdown(
        f"""
        <div style="
            background-color: #1E1E1E;
            padding: 30px;
            border-radius: 20px;
            border-left: 8px solid {status_color};
            box-shadow: 0px 4px 20px rgba(0, 0, 0, .25);
            margin-top: 20px;
        ">
            <h2 style="text-align: center; color: white;">
                💰 Predicted Daily Sales
            </h2>
            <h1 style="
                text-align: center;
                color: #00E676;
                font-size: 55px;
                margin: 10px 0;
            ">
                € {prediction:,.2f}
            </h1>
            <h3 style="
                text-align: center;
                color: {status_color};
                margin: 0;
            ">
                {sales_status}
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="🤖 Model",      value="LightGBM")
    with m2:
        st.metric(label="📈 R² Score",   value=f"{meta['r2']:.4f}")
    with m3:
        st.metric(label="📊 RMSE",       value=f"€{meta['rmse_orig']:,.0f}")

    st.divider()

    with st.expander("📑 Model Information"):
        st.markdown(f"""
### Model Pipeline

- **StandardScaler** — included for pipeline interface consistency; LightGBM
  is tree-based and invariant to feature scaling, but the scaler ensures the
  pipeline remains drop-in compatible if a distance-based model is substituted
  in the future.
- **LightGBM Regressor** — Light Gradient Boosting Machine, selected over
  XGBoost based on higher R² ({meta['r2']}) and lower MAE (€{meta['mae_orig']:,.0f}).

### Features ({len(FEATURES)} total)

The prediction uses **{len(FEATURES)} engineered features** across four groups:

| Group | Features |
|---|---|
| Store | Store_TargetEnc, StoreType, Assortment |
| Calendar | Year, Week, DayOfWeek, IsWeekend, IsMonthStart, IsMonthEnd |
| Promotion | Promo, Promo2, Promo2ActiveWeeks, IsPromo2Active, SchoolHoliday, StateHoliday |
| Competition | CompetitionDistance, CompetitionDistanceMissing, CompetitionOpenMissing, CompetitionOpenMonths |

### Training Metrics

| Scale | MAE | RMSE | R² |
|---|---|---|---|
| Log scale | 0.1299 | 0.1813 | {meta['r2']} |
| Original (€) | €{meta['mae_orig']:,.0f} | €{meta['rmse_orig']:,.0f} | — |

### Target Transform

Sales were log-transformed during training using `np.log1p(Sales)`.  
Predictions are converted back with `np.expm1(prediction)` before display.
""")



# ==========================================
# PREDICTION WORKFLOW
# ==========================================

if predict:

    # --- 1. Validate inputs ---
    validation_warnings = validate_inputs(
        store_target_enc=Store_TargetEnc,
        competition_distance=CompetitionDistance,
        competition_open_months=CompetitionOpenMonths,
        promo2_active_weeks=Promo2ActiveWeeks,
        week=int(Week),
        year=int(Year),
    )

    if validation_warnings:
        for warning_msg in validation_warnings:
            st.warning(f"⚠️ {warning_msg}")
        st.info(
            "The prediction will still run, but results may be less reliable "
            "for inputs outside the training distribution."
        )

    # --- 2. Build input DataFrame in training column order ---
    input_data = build_input_dataframe({
        "Store_TargetEnc":          Store_TargetEnc,
        "DayOfWeek":                DayOfWeek,
        "CompetitionDistance":      CompetitionDistance,
        "CompetitionDistanceMissing": CompetitionDistanceMissing,
        "CompetitionOpenMissing":   CompetitionOpenMissing,
        "StateHoliday":             StateHoliday,
        "SchoolHoliday":            SchoolHoliday,
        "Promo":                    Promo,
        "Promo2":                   Promo2,
        "StoreType":                StoreType,
        "Assortment":               Assortment,
        "Year":                     Year,
        "Week":                     Week,
        "IsWeekend":                IsWeekend,
        "IsMonthStart":             IsMonthStart,
        "IsMonthEnd":               IsMonthEnd,
        "CompetitionOpenMonths":    CompetitionOpenMonths,
        "Promo2ActiveWeeks":        Promo2ActiveWeeks,
        "IsPromo2Active":           IsPromo2Active,
    })

    # --- 3. Progress feedback & inference ---
    progress = st.progress(0, text="Preparing data…")
    progress.progress(40, text="Running model…")

    try:
        prediction = run_prediction(input_data)
    except Exception as exc:
        st.error(f"❌ Prediction failed: {exc}")
        st.stop()

    progress.progress(100, text="Prediction complete ✅")

    # --- 4. Display results ---
    display_result(prediction, input_data)

