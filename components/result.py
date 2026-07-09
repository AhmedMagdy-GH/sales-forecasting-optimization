

import pandas as pd
import streamlit as st

# ==========================================
# RESULT DISPLAY
# ==========================================

def get_sales_status(prediction: float) -> tuple[str, str]:
    """Return (status_label, hex_color) based on the predicted sales value."""
    if prediction < 3_000:
        return "🔴 Low Sales Expected",      "#dc3545"
    elif prediction < 7_000:
        return "🟡 Moderate Sales Expected", "#ffc107"
    else:
        return "🟢 High Sales Expected",     "#28a745"


def display_result(prediction: float, input_df: pd.DataFrame, meta: dict, features: list) -> None:
    """Render the full results section: input summary, prediction card, metrics."""
    sales_status, status_color = get_sales_status(prediction)

    st.divider()
    st.subheader("📋 Input Summary")
    st.dataframe(input_df, use_container_width=True)

    st.divider()
    st.markdown(
        f"""
        <div style="
            background-color: #1E1E1E;
            padding: 30px;
            border-radius: 20px;
            border-left: 8px solid {status_color};
            box-shadow: 0px 4px 20px rgba(0, 0, 0, .25);
            margin-top: 20px;
        ">
            <h2 style="text-align: center; color: white;">
                💰 Predicted Daily Sales
            </h2>
            <h1 style="
                text-align: center;
                color: #00E676;
                font-size: 55px;
                margin: 10px 0;
            ">
                € {prediction:,.2f}
            </h1>
            <h3 style="
                text-align: center;
                color: {status_color};
                margin: 0;
            ">
                {sales_status}
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="🤖 Model",      value="XGBoost")
    with m2:
        st.metric(label="📈 R² Score",   value=f"{meta['r2']:.4f}")
    with m3:
        st.metric(label="📊 RMSE",       value=f"€{meta['rmse_orig']:,.0f}")
    with m4:
        st.metric(label="📉 MAPE",       value=f"{meta['mape']:.2%}")

    st.divider()

    with st.expander("📑 Model Information"):
        st.markdown(f"""
### Model Pipeline

- **StandardScaler** — included for pipeline interface consistency; XGBoost is tree-based 
  and therefore invariant to feature scaling, but scaling is retained to keep the pipeline 
  drop-in compatible with distance-based models that might be substituted later.
- **XGBoost Regressor** — Extreme Gradient Boosting, selected as the final forecasting model
  due to its strong predictive performance ({meta['r2']}) and lower MAE (€{meta['mae_orig']:,.0f}).

### Features ({len(features)} total)

The prediction uses **{len(features)} engineered features** across four groups:

| Group | Features |
|---|---|
| Store | Store_TargetEnc, StoreType, Assortment |
| Calendar | Year, Week, DayOfWeek, IsWeekend, IsMonthStart, IsMonthEnd |
| Promotion | Promo, Promo2, Promo2ActiveWeeks, IsPromo2Active, SchoolHoliday, StateHoliday |
| Competition | CompetitionDistance, CompetitionDistanceMissing, CompetitionOpenMissing, CompetitionOpenMonths |

### Training Metrics

| Scale | MAE | RMSE | R² |
|---|---|---|---|
| Log scale | 0.1285 | 0.1792 | {meta['r2']} |
| Original (€) | €{meta['mae_orig']:,.0f} | €{meta['rmse_orig']:,.0f} | — |

### Target Transform

Sales were log-transformed during training using `np.log1p(Sales)`.  
Predictions are converted back with `np.expm1(prediction)` before display.
""")
