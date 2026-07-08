import pandas as pd
import streamlit as st
from datetime import datetime

from config import HISTORY_PATH
from components.result import get_sales_status

# ==========================================
# Prediction Log
# ==========================================


def save_prediction(inputs: dict, prediction: float) -> None:
    """
    Save a single successful prediction to the history CSV.

    Stores:
    - Timestamp (datetime.now())
    - Every model feature/input used for the prediction
    - The predicted sales value
    - The derived sales status/category

    Creates the history file automatically if it does not exist,
    and appends new rows to it otherwise, without re-reading the
    existing file contents.
    """
    sales_status, _ = get_sales_status(prediction)

    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **inputs,
        "Predicted Sales (€)": round(prediction, 2),
        "Sales Level": sales_status,
    }

    new_record = pd.DataFrame([record])

    file_exists = HISTORY_PATH.exists()

    new_record.to_csv(
        HISTORY_PATH,
        mode="a",
        header=not file_exists,
        index=False,
    )


def load_history() -> pd.DataFrame:

    if HISTORY_PATH.exists():

        return pd.read_csv(HISTORY_PATH)

    return pd.DataFrame()


# ==========================================
# PREDICTION HISTORY DASHBOARD
# ==========================================

def render_history() -> None:
    """Render the Prediction History dashboard section."""

    st.divider()
    st.subheader("📜 Prediction History")
    st.caption("Showing the latest 20 predictions.")

    history = load_history()

    if history.empty:
        st.info("No prediction history available.")
        return

    # --------------------------------------
    # Metric Cards
    # --------------------------------------

    predictions = history["Predicted Sales (€)"]

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Total Predictions", history.shape[0])
    with m2:
        st.metric("Average Prediction", f"€{predictions.mean():,.2f}")
    with m3:
        st.metric("Highest Prediction", f"€{predictions.max():,.2f}")
    with m4:
        st.metric("Lowest Prediction", f"€{predictions.min():,.2f}")
        
    st.write("")
    # --------------------------------------
    # Search Box
    # --------------------------------------

    search_term = st.text_input(
        "🔍 Search History",
        placeholder="Search by date, status, week or year..."
    )

    filtered_history = history.copy()

    if search_term:
        search_term_lower = search_term.strip().lower()

        search_columns = ["Timestamp", "Sales Level", "Week", "Year"]
        available_columns = [c for c in search_columns if c in filtered_history.columns]

        mask = pd.Series(False, index=filtered_history.index)
        for col in available_columns:
            mask |= filtered_history[col].astype(str).str.lower().str.contains(
                search_term_lower, na=False
            )

        filtered_history = filtered_history[mask]

    # --------------------------------------
    # Table Display
    # --------------------------------------

    display_columns = [
        "Timestamp",
        "Predicted Sales (€)",
        "Sales Level",
        "Year",
        "Week",
        "Promo",
    ]
    available_display_columns = [c for c in display_columns if c in filtered_history.columns]

    table_data = (
        filtered_history
        .sort_values("Timestamp", ascending=False)
        .head(20)
        [available_display_columns]
        .copy()
    )
    if "Promo" in table_data.columns:
        table_data["Promo"] = table_data["Promo"].map({
            1: "Yes",
            0: "No"
    })
    if "Predicted Sales (€)" in table_data.columns:
            table_data["Predicted Sales (€)"] = (
            table_data["Predicted Sales (€)"]
            .map(lambda x: f"€ {x:,.2f}")
        )

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True,
    )
    
    st.caption(
        f"Showing {len(table_data)} of {len(history)} predictions."
    )

    # --------------------------------------
    # Action Buttons (placeholders only)
    # --------------------------------------

    b1, b2 = st.columns(2)

    with b1:
        st.button("📥 Download CSV", use_container_width=True, disabled=True)

    with b2:
        st.button("🗑 Clear History", use_container_width=True, disabled=True)