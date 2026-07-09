import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monitoring import PredictionLogger, compute_drift, compute_performance, recent_alerts
from alerting import run_all_checks, MAPE_ALERT_THRESHOLD, RMSE_ALERT_THRESHOLD, PSI_ALERT_THRESHOLD

st.set_page_config(page_title="Model Monitoring — Rossmann", page_icon="🩺", layout="wide")

st.title("🩺 Model Monitoring Dashboard")
st.caption("Prediction volume, input drift, rolling accuracy, and alert history for the "
           "Rossmann Sales Forecasting model.")

logger = PredictionLogger()
n_total = logger.count()

if n_total == 0:
    st.info("No predictions logged yet. Generate a forecast on the main page first, "
            "or run `python src/batch_predict.py` to populate the monitoring store.")
    st.stop()

# ---------------------------------------------------------------------------
# Top-line KPIs
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Predictions Logged", f"{n_total:,}")

recent = logger.recent(limit=1000)
recent["timestamp"] = pd.to_datetime(recent["timestamp"])
labeled = recent.dropna(subset=["actual_sales"])
col2.metric("Predictions with Feedback", f"{len(labeled):,}")

perf = compute_performance(window=1000)
if perf:
    col3.metric("Live MAPE", f"{perf['mape']:.1%}",
                delta=f"{(perf['mape'] - 0.1370):+.1%} vs training",
                delta_color="inverse")
    col4.metric("Live RMSE", f"€{perf['rmse']:,.0f}",
                delta=f"€{(perf['rmse'] - 1150.54):+,.0f} vs training",
                delta_color="inverse")
else:
    col3.metric("Live MAPE", "n/a — no feedback yet")
    col4.metric("Live RMSE", "n/a — no feedback yet")

st.divider()

# ---------------------------------------------------------------------------
# Manual re-check
# ---------------------------------------------------------------------------
if st.button("🔄 Run drift & performance checks now"):
    run_all_checks()
    st.success("Checks complete — see Alerts below.")
    st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Prediction volume over time
# ---------------------------------------------------------------------------
st.subheader("📈 Prediction Volume Over Time")
volume = recent.set_index("timestamp").resample("h")["prediction"].count().rename("predictions")
st.bar_chart(volume)

st.divider()

# ---------------------------------------------------------------------------
# Input drift (PSI)
# ---------------------------------------------------------------------------
st.subheader("📊 Input Drift (Population Stability Index)")
st.caption(
    f"PSI < {0.10} = stable · {0.10}–{PSI_ALERT_THRESHOLD} = moderate drift (warning) · "
    f"≥ {PSI_ALERT_THRESHOLD} = significant drift (critical). Computed against the training "
    "reference distribution in `src/reference_stats.json`."
)
window_ceiling = min(1000, n_total)
if window_ceiling <= 20:
    window = window_ceiling
    st.caption(f"Using all {window} logged prediction(s) — log more than 20 for an adjustable window.")
else:
    window = st.slider("Comparison window (most recent N predictions)", 20, window_ceiling,
                        value=min(200, window_ceiling))
drift_df = compute_drift(window=window)

if drift_df.empty:
    st.info("Not enough data to compute drift yet.")
else:
    def _color(row):
        color = {"ok": "#0b3d1e", "warning": "#4d3b00", "critical": "#4d0b0b"}[row["status"]]
        return [f"background-color: {color}"] * len(row)

    st.dataframe(drift_df.style.apply(_color, axis=1), use_container_width=True,
                 hide_index=True)
    st.bar_chart(drift_df.set_index("feature")["psi"])

st.divider()

# ---------------------------------------------------------------------------
# Rolling accuracy over time (only if feedback available)
# ---------------------------------------------------------------------------
st.subheader("🎯 Rolling Accuracy (requires logged feedback)")
if labeled.empty:
    st.info(
        "No ground-truth sales have been logged yet. Use the **Log Actual Sales** "
        "expander on the main forecasting page to close the feedback loop — once actual "
        "sales are recorded, MAE/RMSE/MAPE trends will appear here."
    )
else:
    labeled = labeled.copy()
    labeled["abs_error"] = (labeled["prediction"] - labeled["actual_sales"]).abs()
    labeled["ape"] = labeled["abs_error"] / labeled["actual_sales"].replace(0, pd.NA)
    daily = labeled.set_index("timestamp").resample("D").agg(
        mae=("abs_error", "mean"), mape=("ape", "mean"), n=("prediction", "count")
    ).dropna()
    st.line_chart(daily[["mae"]])
    st.line_chart(daily[["mape"]])
    st.dataframe(labeled[["id", "timestamp", "prediction", "actual_sales", "abs_error"]]
                 .sort_values("timestamp", ascending=False).head(50),
                 use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Alert history
# ---------------------------------------------------------------------------
st.subheader("🚨 Alert History")
alerts = recent_alerts(limit=100)
if alerts.empty:
    st.success("No alerts have been raised.")
else:
    def _sev_color(row):
        color = {"info": "#0b2e4d", "warning": "#4d3b00", "critical": "#4d0b0b"}.get(row["severity"], "")
        return [f"background-color: {color}"] * len(row)
    st.dataframe(alerts.style.apply(_sev_color, axis=1), use_container_width=True, hide_index=True)

st.caption(
    f"Alert thresholds — MAPE > {MAPE_ALERT_THRESHOLD:.0%}, RMSE > €{RMSE_ALERT_THRESHOLD:,.0f}, "
    f"PSI ≥ {PSI_ALERT_THRESHOLD}. Configure via environment variables "
    "(MAPE_ALERT_THRESHOLD, RMSE_ALERT_THRESHOLD, PSI_ALERT_THRESHOLD) and optionally "
    "SLACK_WEBHOOK_URL / SMTP_* for external notifications — see `src/alerting.py`."
)
