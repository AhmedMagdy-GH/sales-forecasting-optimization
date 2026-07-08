

import streamlit as st

# ==========================================
# SIDEBAR
# ==========================================

def render_sidebar(meta: dict) -> None:
    """Render the sidebar exactly as in the original app, using meta for all model info."""
    with st.sidebar:

        st.title("🏪 Rossmann")
        st.caption("Sales Forecasting Dashboard")
        st.divider()

        st.success("Machine Learning Model")
        st.markdown(f"""
**{meta['model']}**

**Pipeline**

- StandardScaler
- XGBoost Regressor
""")

        st.divider()
        st.success("Technologies")
        st.markdown("""
- Python
- Streamlit
- XGBoost
- Pandas
- NumPy
- Scikit-Learn
- Joblib
""")

        st.divider()
        st.success("Performance")
        st.metric("R² Score", f"{meta['r2']:.4f}")
        st.metric("MAE",      f"€{meta['mae_orig']:,.0f}")
        st.metric("RMSE",     f"€{meta['rmse_orig']:,.0f}")
        st.metric("MAPE",     f"{meta['mape']:.2%}")
