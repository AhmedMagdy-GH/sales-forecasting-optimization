# 1. Problem Statement & Objectives

## 🧩 Problem Statement
Retail and e-commerce businesses struggle to **anticipate future demand**, which leads to:

- **Stock-outs** that cost revenue and customer trust.
- **Over-stocking** that ties up capital and increases storage costs.
- **Reactive marketing & staffing** decisions that miss seasonal opportunities.
- **Manual, spreadsheet-driven forecasts** that ignore promotions, holidays and weather effects.

There is a clear need for an automated, data-driven system that converts historical sales data into
**accurate, explainable, short- and medium-term sales forecasts** accessible to non-technical
business users.

## 🎯 Project Goals
1. **Forecast daily/weekly sales** at SKU-store granularity using an **XGBoost** regression model
   on engineered time-series features.
2. Achieve **MAPE ≤ 10 %** on the hold-out test set (target metric).
3. Provide a **Streamlit web app** where business users can:
   - Upload new data or pick a store/product.
   - Generate forecasts for a chosen horizon (7 / 14 / 30 / 90 days).
   - Compare scenarios (with/without promotion, holiday).
   - Download forecasts as CSV / Excel.
4. Track all experiments, parameters and metrics with **MLflow**.
5. Version raw and processed data with **DVC** (Google Drive remote).
6. Continuously **monitor model drift** and retrain on a schedule.
7. Deploy to a **containerized cloud environment** (Docker → Streamlit Cloud / AWS / GCP).

## 📦 In Scope
- Univariate + multivariate time-series forecasting.
- Feature engineering (calendar, lags, rolling windows, promo flags).
- Model training, tuning, evaluation, registry, deployment.
- Interactive UI for forecast generation and visualization.

## 🚫 Out of Scope
- Real-time streaming ingestion (batch / on-demand only).
- Pricing optimization (only demand forecasting is delivered).
- Mobile native apps.

