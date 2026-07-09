import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import FEATURES, DEFAULT_MODEL_NAME
from monitoring import PredictionLogger, log_feedback

st.set_page_config(page_title="Batch Predictions — Rossmann", page_icon="📦", layout="wide")


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model_path = "src/model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ **model.pkl not found.** Make sure you launch Streamlit from the project root.")
        st.stop()
    return joblib.load(model_path)


st.title("📦 Batch Sales Forecasting")
st.caption("Upload a CSV of many store/day rows and generate sales forecasts for all of them at once — "
           "no command line needed.")

with st.expander("📋 Required CSV columns", expanded=False):
    st.write(
        "Your file must contain these 19 columns (extra columns are fine — they'll be kept in the output "
        "but ignored by the model):"
    )
    st.code(", ".join(FEATURES))
    template = pd.DataFrame([{
        "Store_TargetEnc": 8.50, "DayOfWeek": 5, "CompetitionDistance": 500.0,
        "CompetitionDistanceMissing": 0, "CompetitionOpenMissing": 0, "StateHoliday": 0,
        "SchoolHoliday": 0, "Promo": 1, "Promo2": 0, "StoreType": 0, "Assortment": 0,
        "Year": 2015, "Week": 30, "IsWeekend": 0, "IsMonthStart": 0, "IsMonthEnd": 0,
        "CompetitionOpenMonths": 12.0, "Promo2ActiveWeeks": 0.0, "IsPromo2Active": 0,
    }])
    st.download_button(
        "⬇️ Download a template CSV",
        data=template.to_csv(index=False),
        file_name="batch_predict_template.csv",
        mime="text/csv",
    )

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

log_to_monitoring = st.checkbox(
    "Log these predictions to the Monitoring Dashboard", value=True,
    help="Uncheck this for a quick test run you don't want showing up in drift/performance tracking."
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"❌ Could not read that file as CSV: {exc}")
        st.stop()

    st.subheader("📄 Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"{len(df):,} row(s) loaded.")

    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required column(s): {', '.join(missing_cols)}")
        st.stop()

    non_numeric = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        st.warning(
            f"⚠️ These columns aren't numeric and will be coerced (non-numeric values become blank / dropped "
            f"rows): {', '.join(non_numeric)}"
        )

    st.divider()
    run = st.button("🚀 Run Batch Predictions", use_container_width=True)

    if run:
        work_df = df.copy()
        for c in FEATURES:
            work_df[c] = pd.to_numeric(work_df[c], errors="coerce")

        bad_rows = work_df[FEATURES].isna().any(axis=1)
        if bad_rows.any():
            st.warning(f"⚠️ Dropping {int(bad_rows.sum())} row(s) with invalid/missing values in required columns.")
            work_df = work_df[~bad_rows].reset_index(drop=True)

        if work_df.empty:
            st.error("❌ No valid rows left to predict on.")
            st.stop()

        progress = st.progress(0, text="Loading model…")
        model = load_model()
        progress.progress(40, text="Running predictions…")

        X = work_df[FEATURES]
        log_preds = model.predict(X)
        work_df["PredictedSales"] = np.expm1(log_preds)

        progress.progress(80, text="Logging results…")
        if log_to_monitoring:
            logger = PredictionLogger(model_version=DEFAULT_MODEL_NAME)
            for _, row in work_df.iterrows():
                logger.log(row[FEATURES].to_dict(), float(row["PredictedSales"]))

        progress.progress(100, text="Done ✅")
        progress.empty()

        st.success(f"Generated {len(work_df):,} predictions.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows Predicted", f"{len(work_df):,}")
        c2.metric("Average Predicted Sales", f"€{work_df['PredictedSales'].mean():,.0f}")
        c3.metric("Total Predicted Sales", f"€{work_df['PredictedSales'].sum():,.0f}")

        st.subheader("📊 Results")
        st.dataframe(work_df, use_container_width=True)
        st.bar_chart(work_df["PredictedSales"])

        st.download_button(
            "⬇️ Download predictions as CSV",
            data=work_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if log_to_monitoring:
            st.caption("✅ These predictions were logged — check the Monitoring Dashboard page.")
else:
    st.info("Upload a CSV to get started, or download the template above.")

st.divider()

# ---------------------------------------------------------------------------
# Bulk feedback — log many actual-sales values at once
# ---------------------------------------------------------------------------
st.subheader("📝 Bulk Log Actual Sales (feedback loop)")
st.caption(
    "Instead of entering actual sales one at a time on the main page, upload a CSV with two "
    "columns — a **Prediction ID** and its **Actual Sales** — to log feedback for many "
    "predictions in one click. This is what feeds rolling MAE/RMSE/MAPE on the Monitoring Dashboard."
)

fb_template = pd.DataFrame([
    {"PredictionID": 1, "ActualSales": 5230.00},
    {"PredictionID": 2, "ActualSales": 2185.50},
])
st.download_button(
    "⬇️ Download a feedback template CSV",
    data=fb_template.to_csv(index=False),
    file_name="feedback_template.csv",
    mime="text/csv",
)

feedback_file = st.file_uploader("Upload feedback CSV", type=["csv"], key="feedback_uploader")

# Accept a few common header spellings so this isn't fussy about exact casing/underscores
ID_ALIASES = {"predictionid", "prediction_id", "id"}
SALES_ALIASES = {"actualsales", "actual_sales", "actual", "sales"}

if feedback_file is not None:
    try:
        fb_df = pd.read_csv(feedback_file)
    except Exception as exc:
        st.error(f"❌ Could not read that file as CSV: {exc}")
        st.stop()

    normalized_cols = {c.lower().replace(" ", ""): c for c in fb_df.columns}
    id_col = next((normalized_cols[a] for a in ID_ALIASES if a in normalized_cols), None)
    sales_col = next((normalized_cols[a] for a in SALES_ALIASES if a in normalized_cols), None)

    if id_col is None or sales_col is None:
        st.error(
            "❌ Couldn't find the required columns. Your CSV needs a Prediction ID column "
            "(e.g. `PredictionID`) and an Actual Sales column (e.g. `ActualSales`). "
            f"Found columns: {list(fb_df.columns)}"
        )
        st.stop()

    fb_df = fb_df[[id_col, sales_col]].rename(columns={id_col: "PredictionID", sales_col: "ActualSales"})
    fb_df["PredictionID"] = pd.to_numeric(fb_df["PredictionID"], errors="coerce")
    fb_df["ActualSales"] = pd.to_numeric(fb_df["ActualSales"], errors="coerce")

    invalid = fb_df[["PredictionID", "ActualSales"]].isna().any(axis=1)
    if invalid.any():
        st.warning(f"⚠️ Skipping {int(invalid.sum())} row(s) with a missing/non-numeric ID or sales value.")
        fb_df = fb_df[~invalid]

    st.dataframe(fb_df, use_container_width=True, hide_index=True)

    if st.button("📥 Log All Feedback", use_container_width=True):
        n_ok, not_found = 0, []
        for _, row in fb_df.iterrows():
            pid = int(row["PredictionID"])
            found = log_feedback(pid, float(row["ActualSales"]))
            if found:
                n_ok += 1
            else:
                not_found.append(pid)

        if n_ok:
            st.success(f"✅ Logged feedback for {n_ok} prediction(s).")
        if not_found:
            st.error(
                f"❌ {len(not_found)} prediction ID(s) didn't match anything in the monitoring "
                f"log and were skipped: {not_found}"
            )
        if n_ok:
            st.caption("Check the Monitoring Dashboard page — Live MAPE/RMSE and the rolling accuracy "
                       "charts should now reflect this feedback.")
