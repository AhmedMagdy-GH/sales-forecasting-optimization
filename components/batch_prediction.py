import numpy as np
import pandas as pd
import streamlit as st

from config import FEATURES
from components.result import get_sales_status
from utils.prediction import build_input_dataframe, run_prediction

# ==========================================
# BATCH PREDICTION — Upload & Processing
# ==========================================

def render_batch_prediction(model) -> None:
    
    """Render the Batch Prediction upload interface and run batch inference."""

    st.divider()
    st.subheader("📂 Batch Prediction")
    st.caption("Upload a CSV file for batch prediction.")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    batch_df = None

    if uploaded_file is not None:

        try:
            batch_df = pd.read_csv(uploaded_file)

            st.success(f"✅ Number of rows loaded: {len(batch_df)}")

            st.dataframe(
                batch_df.head(10),
                use_container_width=True,
                hide_index=True,
            )

        except Exception as exc:
            st.error(f"❌ Failed to read CSV: {exc}")
            batch_df = None

    run_batch = st.button(
        "🚀 Run Batch Prediction",
        use_container_width=True,
        disabled=batch_df is None,
    )

    if run_batch and batch_df is not None:

        # --- 1. Validate required columns ---
        missing_columns = [col for col in FEATURES if col not in batch_df.columns]

        if missing_columns:
            st.error(
                "❌ The uploaded CSV is missing required columns: "
                + ", ".join(missing_columns)
            )
            return

        

        # Reuse the same column ordering enforced by build_input_dataframe,
        # applied here across the whole batch rather than row-by-row so we
        # avoid looping over run_prediction() for every row.
        input_data = batch_df[FEATURES]

        try:
            if len(input_data) == 1:
                # Reuse run_prediction() directly for the single-row case.
                predictions = [run_prediction(model, input_data)]
            else:
                # Vectorized prediction across the whole DataFrame, using the
                # same inverse log-transform (np.expm1) applied by run_prediction().
                log_predictions = model.predict(input_data)
                predictions = np.expm1(log_predictions)
        except Exception as exc:
            st.error(f"❌ Batch prediction failed: {exc}")
            return

        # --- 3. Append prediction columns ---
        result_df = batch_df.copy()
        result_df["Predicted Sales (€)"] = (
            pd.Series(predictions)
            .round(2)
        )
        result_df["Sales Level"] = result_df["Predicted Sales (€)"].apply(
            lambda value: get_sales_status(value)[0]
        )

        # --- 4. Display results ---
        st.success("✅ Batch prediction completed successfully.")

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
        )