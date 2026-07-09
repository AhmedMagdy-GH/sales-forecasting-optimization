

import streamlit as st

# ==========================================
# FOOTER
# ==========================================

def render_footer(meta: dict) -> None:
    """Render the footer exactly as in the original app."""
    st.divider()

    footer_left, footer_center, footer_right = st.columns(3)

    with footer_left:
        st.info("""
### 🏪 Project

Rossmann Sales Forecasting

Machine Learning Deployment
""")

    with footer_center:
        st.info("""
### ⚙️ Technologies

- Python
- Streamlit
- XGBoost
- Scikit-Learn
- Pandas
""")

    with footer_right:
        st.info(f"""
### 📊 Model Performance

R² : {meta['r2']:.4f}

MAE : €{meta['mae_orig']:,.0f}

RMSE : €{meta['rmse_orig']:,.0f}

MAPE : {meta['mape']:.2%}
""")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
<div style="
    background: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border-top: 4px solid #00E676;
">
    <h3 style="color: white;">
        🏪 Rossmann Sales Forecasting Dashboard
    </h3>
    <p style="color: #CFCFCF; font-size: 18px;">
        Graduation Project
    </p>
    <p style="color: #9E9E9E;">
        Machine Learning Deployment using Streamlit
    </p>
    <p style="color: #00E676; font-weight: bold;">
        XGBoost • Python • Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
