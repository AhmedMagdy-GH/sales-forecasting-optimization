import streamlit as st

from components.history import load_history
from components.feedback import load_feedback
from components.monitoring import load_performance_log
from components.drift import compute_drift_report

# ==========================================
# ALERT SYSTEM
# ==========================================
#
# Generates real alerts from the application's own CSV logs, reusing
# the exact loaders/computations already used by the other dashboards
# (Prediction History, Model Monitoring, Model Drift Detection, and
# Feedback Loop). No placeholder values, no independent CSV reads.

DRIFT_THRESHOLD = 25.0            # Drift (%) >= this -> High Drift alert
SUCCESS_RATE_THRESHOLD = 95.0     # Success Rate < this -> Low Success Rate alert
INFERENCE_TIME_THRESHOLD_MS = 500.0  # Average Inference Time (ms) > this -> High Latency alert
POSITIVE_FEEDBACK_THRESHOLD = 80.0   # Positive Feedback Rate < this -> Low Feedback alert

DRIFT_MIN_SAMPLES = 5  # keep consistent with components.drift.MIN_SAMPLES


def _check_drift_alert() -> dict | None:
    """High Model Drift alert, based on components.drift.compute_drift_report()."""
    history = load_history().tail(100)

    if history.empty or len(history) < DRIFT_MIN_SAMPLES:
        return None

    drift_report = compute_drift_report(history)

    if drift_report.empty:
        return None

    max_drift = drift_report["Drift (%)"].max()

    if max_drift >= DRIFT_THRESHOLD:
        return {
            "level": "error",
            "icon": "🔴",
            "title": "High Model Drift Detected",
            "message": (
                "Current prediction inputs differ significantly from the "
                "training distribution. Model retraining is recommended."
            ),
        }

    return None


def _check_success_rate_alert() -> dict | None:
    """Low Prediction Success Rate alert, based on components.monitoring.load_performance_log()."""
    log = load_performance_log().tail(100)

    if log.empty or "Status" not in log.columns:
        return None

    total_predictions = len(log)
    successful_predictions = (log["Status"] == "Success").sum()
    success_rate = (successful_predictions / total_predictions) * 100

    if success_rate < SUCCESS_RATE_THRESHOLD:
        return {
            "level": "warning",
            "icon": "🟡",
            "title": "Low Prediction Success Rate",
            "message": (
                "Prediction failures are increasing. "
                "Please inspect the deployment logs."
            ),
        }

    return None


def _check_inference_time_alert() -> dict | None:
    """High Inference Latency alert, based on components.monitoring.load_performance_log()."""
    log = load_performance_log().tail(100)

    if log.empty or "Execution Time (ms)" not in log.columns:
        return None

    avg_execution_time = log["Execution Time (ms)"].mean()

    if avg_execution_time > INFERENCE_TIME_THRESHOLD_MS:
        return {
            "level": "warning",
            "icon": "🟠",
            "title": "High Inference Latency",
            "message": (
                "Average model inference time exceeds the recommended threshold."
            ),
        }

    return None


def _check_feedback_alert() -> dict | None:
    """Low User Feedback alert, based on components.feedback.load_feedback()."""
    feedback = load_feedback().tail(100)
    if feedback.empty or "Feedback" not in feedback.columns:
        return None

    total_feedback = len(feedback)
    correct_feedback = (feedback["Feedback"] == "Correct").sum()
    positive_rate = (correct_feedback / total_feedback) * 100

    if positive_rate < POSITIVE_FEEDBACK_THRESHOLD:
        return {
            "level": "error",
            "icon": "🔴",
            "title": "Low User Feedback",
            "message": (
                "Users report poor prediction quality. "
                "Consider retraining the forecasting model."
            ),
        }

    return None


def get_active_alerts() -> list:
    """Collect every active alert from the real, already-logged application data."""
    checks = (
        _check_drift_alert,
        _check_success_rate_alert,
        _check_inference_time_alert,
        _check_feedback_alert,
    )

    return [alert for alert in (check() for check in checks) if alert is not None]


# ==========================================
# ACTIVE ALERTS SECTION
# ==========================================

def render_alerts() -> None:
    """Render the Active Alerts section."""

    st.divider()
    st.subheader("🚨 Active Alerts")

    alerts = get_active_alerts()

    if not alerts:
        st.success("🟢 No active alerts.")
        return

    for alert in alerts:
        st.markdown(f"**{alert['icon']} {alert['title']}**")

        if alert["level"] == "error":
            st.error(alert["message"])
        elif alert["level"] == "warning":
            st.warning(alert["message"])
        else:
            st.info(alert["message"])