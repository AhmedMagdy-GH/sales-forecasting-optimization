# 14. Wireframes & Mockups (Streamlit Screens)

The Streamlit app is organized as a **multi-page** application with a left sidebar
for navigation and global filters.

```
┌────────────────────────────────────────────────────────────────────────────┐
│ 🛒 Sales Forecasting & Optimization               👤 user ▾   ⚙ Settings   │
├──────────────┬─────────────────────────────────────────────────────────────┤
│              │                                                             │
│  NAV         │           <PAGE CONTENT>                                    │
│  ─────       │                                                             │
│  🏠 Home     │                                                             │
│  📥 Upload   │                                                             │
│  🔎 EDA      │                                                             │
│  📈 Forecast │                                                             │
│  🧪 Scenarios│                                                             │
│  📊 Monitoring│                                                            │
│  🛠 Admin    │                                                             │
│              │                                                             │
│  Filters     │                                                             │
│  ─────       │                                                             │
│  Store: [▾]  │                                                             │
│  SKU:   [▾]  │                                                             │
│  Range:[📅]  │                                                             │
└──────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 14.1 🏠 Home / Dashboard

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Welcome back, Ahmed 👋                                                   │
│                                                                          │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐             │
│  │ Total Sales│ │ Forecast   │ │ MAPE (30d) │ │ Active     │             │
│  │  $1.24 M   │ │  +8.3% ▲   │ │   7.2%     │ │  Model: v9 │             │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘             │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐│
│  │  Sales last 90 days vs forecast (line chart)                         ││
│  │                                                                      ││
│  └──────────────────────────────────────────────────────────────────────┘│
│  ┌─────────────── Top 5 SKUs ──────────────┬──── Alerts (2) ───────────┐│
│  │ SKU 102  ███████████████ 412k           │ ⚠ Drift on Store#7         ││
│  │ SKU 045  ████████████  301k             │ ⚠ Stock-out risk SKU#22    ││
│  │ SKU 077  █████████    240k              │                            ││
│  └────────────────────────────────────────┴────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────┘
```

## 14.2 📥 Upload Data

```
┌────────────────────────────────────────────────────────┐
│  Upload Sales CSV                                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │  ⬆ Drag and drop CSV here, or [Browse files]      │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  Required columns:                                     │
│   ✔ date    ✔ store_id   ✔ sku_id    ✔ units_sold     │
│   ✔ promo   ✔ holiday                                  │
│                                                        │
│  Preview (first 100 rows): [data table]                │
│                                                        │
│  [ Validate ]   [ Save & Version with DVC ]            │
└────────────────────────────────────────────────────────┘
```

## 14.3 🔎 EDA Dashboard

```
┌────────────────────────────────────────────────────────┐
│ Tabs: [Overview] [Trends] [Seasonality] [Correlations] │
│                                                        │
│ Overview                                               │
│  • Rows: 1 248 920  • Date range: 2019-01-01 → 2024-12-31│
│  • Missing: 0.3%   • Duplicates: 0                     │
│                                                        │
│ ┌──────────────── Histogram of units_sold ────────────┐│
│ │                                                     ││
│ └─────────────────────────────────────────────────────┘│
│ ┌──────────────── Correlation heatmap ────────────────┐│
│ │                                                     ││
│ └─────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────┘
```

## 14.4 📈 Forecast

```
┌────────────────────────────────────────────────────────┐
│  Forecast configuration                                │
│  Store: [Cairo-01 ▾]   SKU: [SKU-045 ▾]                │
│  Horizon: ( 7 ) ( 14 ) (•30•) ( 90 )  days             │
│  Promo: [☑]   Holiday: [☐]                             │
│                                                        │
│  [ 🚀 Generate Forecast ]                              │
│                                                        │
│  ┌──────────────── Forecast chart (Plotly) ───────────┐│
│  │  actuals ───  forecast ━━  CI band ░░               ││
│  │                                                    ││
│  └────────────────────────────────────────────────────┘│
│  KPI: Expected revenue $84 200  (±$5 100)              │
│       Suggested re-order qty:  1 240 units             │
│  [ ⬇ Download CSV ] [ ⬇ Download Excel ]               │
└────────────────────────────────────────────────────────┘
```

## 14.5 🧪 Scenarios

```
┌────────────────────────────────────────────────────────┐
│ Compare scenarios side-by-side                         │
│                                                        │
│  Scenario A      Scenario B                            │
│  ─────────       ─────────                             │
│  Promo: ☑         Promo: ☐                             │
│  Holiday: ☐       Holiday: ☑                           │
│                                                        │
│ ┌──────────── Dual-line chart ──────────────────────── │
│ │ A ━━ B ──                                            │
│ └────────────────────────────────────────────────────  │
│ Δ Revenue: +12.4%   Δ Units: +9.8%                     │
└────────────────────────────────────────────────────────┘
```

## 14.6 📊 Monitoring (Admin / ML Engineer)

```
┌────────────────────────────────────────────────────────┐
│ Model: sales_xgb  v9 (Production since 2026-04-01)     │
│                                                        │
│ Rolling MAPE last 30 days:  7.2%   ▼ green             │
│ PSI (feature drift):        0.08   ▼ green             │
│ Last retraining:            2026-04-19                 │
│                                                        │
│ ┌──── Predictions vs actuals ──────────────────────┐   │
│ │                                                  │   │
│ └──────────────────────────────────────────────────┘   │
│ [ 🔄 Trigger retraining ]   [ 📜 View MLflow runs ]    │
└────────────────────────────────────────────────────────┘
```

