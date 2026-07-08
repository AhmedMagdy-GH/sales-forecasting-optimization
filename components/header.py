

import streamlit as st

# ==========================================
# HEADER
# ==========================================

def render_header(meta: dict, features: list):
    """
    Render the page header, caption, divider, and the two-column layout
    (left, right). Renders the right-side Model Summary Cards panel and
    returns the `left` column object so the calling code can render the
    Store Information form inside it, exactly as in the original layout.
    """
    st.title("🏪 Rossmann Sales Forecasting")
    st.caption("Machine Learning Deployment using Streamlit & XGBoost")
    st.divider()

    left, right = st.columns([3, 1])
    # ==========================================
    # RIGHT PANEL — Model Summary Cards
    # ==========================================

    with right:

        st.markdown("### 🤖 Model")

        panel_items = [
            ("Algorithm", "XGBoost"),
            ("Features",  str(len(features))),
            ("R² Score",  f"{meta['r2']:.4f}"),
            ("MAE",       f"€{meta['mae_orig']:,.0f}"),
            ("RMSE",      f"€{meta['rmse_orig']:,.0f}"),
            ("MAPE",      f"{meta['mape']:.2%}")
        ]

        for title, value in panel_items:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">{title}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    return left
