import pandas as pd
import streamlit as st
from datetime import datetime

from config import PERFORMANCE_LOG_PATH

# ==========================================
# Model Performance Monitoring
# ==========================================

MODEL_NAME = "XGBoost"
MODEL_VERSION = "v2"


def log_prediction_performance(
    prediction: float,
    execution_time_ms: float,
    status: str = "Success"
) -> None:
    """
    Append a single performance log record to the performance log CSV.

    Stores:
    - Timestamp (datetime.now())
    - Model Name
    - Model Version
    - Prediction
    - Execution Time (ms)
    - Status

    Creates the performance log file automatically if it does not exist,
    and appends new rows to it otherwise, without re-reading the
    existing file contents.
    """
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model Name": MODEL_NAME,
        "Model Version": MODEL_VERSION,
        "Prediction": round(prediction, 2),
        "Execution Time (ms)": round(execution_time_ms, 2),
        "Status": status,
    }

    new_record = pd.DataFrame([record])

    file_exists = PERFORMANCE_LOG_PATH.exists()

    new_record.to_csv(
        PERFORMANCE_LOG_PATH,
        mode="a",
        header=not file_exists,
        index=False,
    )


def load_performance_log() -> pd.DataFrame:

    if PERFORMANCE_LOG_PATH.exists():

        return pd.read_csv(PERFORMANCE_LOG_PATH)

    return pd.DataFrame()


# ==========================================
# MODEL MONITORING DASHBOARD
# ==========================================

def render_monitoring_dashboard() -> None:
    """Render the Model Monitoring dashboard section."""

    st.divider()
    st.subheader("🤖 Model Monitoring")

    log = load_performance_log()

    if log.empty:
        st.info("No monitoring data available.")
        return

    # --------------------------------------
    # Metric Calculations
    # --------------------------------------

    total_predictions = len(log)

    execution_times = log["Execution Time (ms)"]
    avg_execution_time = execution_times.mean()
    fastest_prediction = execution_times.min()
    slowest_prediction = execution_times.max()

    successful_predictions = (log["Status"] == "Success").sum()
    success_rate = (successful_predictions / total_predictions) * 100
    all_successful = successful_predictions == total_predictions

    # --------------------------------------
    # Metric Cards
    # --------------------------------------

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    
    model_status = "Healthy 🟢" if all_successful else "Attention 🟡"

    with r1c1:
        st.metric(
            "Model Status",
            model_status,
        )
    with r1c2:
        st.metric("Current Model", MODEL_NAME)
    with r1c3:
        st.metric("Model Version", MODEL_VERSION)
    with r1c4:
        st.metric("Total Predictions", f"{total_predictions}")

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)

    with r2c1:
        st.metric("Average Inference Time (ms)", f"{avg_execution_time:.2f}")
    with r2c2:
        st.metric("Fastest Prediction (ms)", f"{fastest_prediction:.2f}")
    with r2c3:
        st.metric("Slowest Prediction (ms)", f"{slowest_prediction:.2f}")
    with r2c4:
        st.metric("Success Rate", f"{success_rate:.2f}%")

    # --------------------------------------
    # Monitoring Summary
    # --------------------------------------

    st.divider()
    st.markdown("### Monitoring Summary")

    if all_successful:
        st.success(
            "🟢 Healthy\n\n"
            "Average inference time is within normal limits.\n\n"
            "No failed predictions detected."
        )
    else:
        failed_predictions = total_predictions - successful_predictions
        st.warning(
            f"🟡 Attention Needed\n\n"
            f"{failed_predictions} failed prediction(s) detected.\n\n"
            f"Current success rate: {success_rate:.2f}%."
        )

    # --------------------------------------
    # Latest Prediction
    # --------------------------------------

    st.divider()
    st.markdown("### Latest Prediction")

    latest_columns = ["Timestamp", "Prediction", "Execution Time (ms)", "Status"]
    available_latest_columns = [c for c in latest_columns if c in log.columns]

    latest_record = (
        log.sort_values("Timestamp", ascending=False)
        .head(1)[available_latest_columns]
    )

    st.dataframe(
        latest_record,
        use_container_width=True,
        hide_index=True,
    )