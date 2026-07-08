import pandas as pd
import streamlit as st

from config import FEATURE_RANGES
from components.history import load_history

# ==========================================
# MODEL DRIFT DETECTION DASHBOARD
# ==========================================
#
# Real drift detection based on actual logged prediction inputs
# (prediction_log.csv, via components.history.load_history) compared
# against the training distribution reference ranges already defined
# in config.FEATURE_RANGES (the same reference used by
# utils.validation.validate_inputs).
#
# No training dataset is bundled with the deployed app, so the
# reference used here is each feature's known training range rather
# than a full reference distribution. A feature's current average is
# considered drift-free while it stays inside that range; drift is
# reported only for how far the average has moved outside the range,
# normalized by the range's width and capped at 100%.

MIN_SAMPLES = 5  # minimum logged predictions required for a meaningful signal

HEALTHY_THRESHOLD = 10.0   # % drift below this -> Healthy
MODERATE_THRESHOLD = 25.0  # % drift below this -> Moderate, otherwise High


def _feature_drift(current_mean: float, lo: float, hi: float) -> float:
    """
    Compute drift (%) of a feature's current mean relative to its
    training reference range.

    - Inside [lo, hi]: no drift (0%).
    - Below lo: how far below the minimum, normalized by the range width.
    - Above hi: how far above the maximum, normalized by the range width.
    - Capped at 100% to keep the dashboard readable.
    """
    range_width = hi - lo
    if range_width <= 0:
        return 0.0

    if current_mean < lo:
        distance = lo - current_mean
    elif current_mean > hi:
        distance = current_mean - hi
    else:
        return 0.0

    drift_pct = distance / range_width * 100
    return min(drift_pct, 100.0)


def _drift_status(drift_pct: float) -> tuple[str, str]:
    """Return (status_label, badge) for a given drift percentage."""
    if drift_pct < HEALTHY_THRESHOLD:
        return "Healthy", "🟢"
    if drift_pct < MODERATE_THRESHOLD:
        return "Moderate", "🟡"
    return "High Drift", "🔴"


def compute_drift_report(history: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-feature drift report comparing the current logged
    prediction inputs against the training reference ranges.

    Returns a DataFrame with columns:
    Feature, Current Mean, Reference Range, Drift (%), Status
    """
    rows = []

    for feature, (lo, hi) in FEATURE_RANGES.items():
        if feature not in history.columns:
            continue

        current_mean = pd.to_numeric(history[feature], errors="coerce").mean()
        if pd.isna(current_mean):
            continue

        drift_pct = _feature_drift(current_mean, lo, hi)
        status_label, badge = _drift_status(drift_pct)

        rows.append({
            "Feature": feature,
            "Current Mean": round(current_mean, 2),
            "Reference Range": f"{lo:,.1f} – {hi:,.1f}",
            "Drift (%)": round(drift_pct, 2),
            "Status": f"{badge} {status_label}",
        })

    return pd.DataFrame(rows)


def render_drift_dashboard() -> None:
    """Render the Model Drift Detection dashboard."""

    st.divider()
    st.subheader("📈 Model Drift Detection")
    st.caption("Current logged prediction inputs compared with training reference ranges.")

    history = load_history()

    if history.empty or len(history) < MIN_SAMPLES:
        logged = 0 if history.empty else len(history)
        st.info(
            f"Not enough logged predictions to assess drift yet "
            f"({logged}/{MIN_SAMPLES} minimum). Run a few more predictions "
            f"to populate the drift dashboard."
        )
        return

    drift_report = compute_drift_report(history)

    if drift_report.empty:
        st.info("No monitored features found in the prediction log.")
        return

    # --------------------------------------
    # Metric Calculations
    # --------------------------------------

    reference_features = len(FEATURE_RANGES)
    features_checked = len(drift_report)
    features_with_drift = (drift_report["Drift (%)"] >= HEALTHY_THRESHOLD).sum()
    max_drift = drift_report["Drift (%)"].max()

    overall_healthy = features_with_drift == 0
    overall_high = (drift_report["Drift (%)"] >= MODERATE_THRESHOLD).any()

    if overall_high:
        overall_status = "High Drift 🔴"
    elif not overall_healthy:
        overall_status = "Moderate 🟡"
    else:
        overall_status = "Healthy 🟢"

    # --------------------------------------
    # Metric Cards
    # --------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Overall Status", overall_status)
    with c2:
        st.metric("Reference Features", reference_features)
    with c3:
        st.metric("Features Checked", features_checked)
    with c4:
        st.metric("Features with Drift", int(features_with_drift))
    with c5:
        st.metric("Maximum Drift", f"{max_drift:.1f}%")

    # --------------------------------------
    # Drift Summary
    # --------------------------------------

    st.markdown("### Drift Summary")

    if overall_high:
        st.error(
            "🔴 High drift detected on one or more features. "
            "Predictions relying on these inputs may be unreliable — "
            "consider investigating input sources or retraining the model."
        )
    elif not overall_healthy:
        st.warning(
            "🟡 Moderate drift detected. Feature distributions are starting "
            "to shift away from the training reference ranges."
        )
    else:
        st.success("No significant feature drift detected.")

    # --------------------------------------
    # Drift Detail Table
    # --------------------------------------

    st.dataframe(
        drift_report.sort_values("Drift (%)", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Computed from {len(history)} logged prediction(s). "
        f"Drift (%) is 0 while a feature's current average stays within "
        f"its training reference range, and otherwise reflects how far "
        f"outside that range it has moved, relative to the range's width "
        f"(capped at 100%)."
    )