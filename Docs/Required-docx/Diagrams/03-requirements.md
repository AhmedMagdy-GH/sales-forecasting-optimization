# 3. Functional & Non-Functional Requirements

## ✅ Functional Requirements (FR)

| ID    | Requirement |
|-------|-------------|
| FR-1  | The system **shall ingest** historical sales data via CSV upload or DVC-tracked file. |
| FR-2  | The system **shall validate** the uploaded schema (date, store_id, sku_id, sales, promo, holiday). |
| FR-3  | The system **shall display** EDA dashboards (trends, seasonality, correlations, missingness). |
| FR-4  | The system **shall engineer** time-series features (lags, rolling means/std, date parts, holiday flags). |
| FR-5  | The system **shall train** an XGBoost regression model with time-series cross-validation. |
| FR-6  | The system **shall tune** hyper-parameters (Optuna / GridSearchCV). |
| FR-7  | The system **shall log** every training run (params, metrics, artifacts) to **MLflow**. |
| FR-8  | The system **shall register** the best model in the MLflow Model Registry. |
| FR-9  | The system **shall generate** sales forecasts for a user-selected store, SKU and horizon (7/14/30/90 days). |
| FR-10 | The system **shall allow** scenario comparison (with vs. without promotion / holiday). |
| FR-11 | The system **shall visualize** forecast vs. actuals with interactive Plotly charts. |
| FR-12 | The system **shall export** forecasts as CSV or Excel. |
| FR-13 | The system **shall support** batch and on-demand prediction modes. |
| FR-14 | The system **shall version** datasets with **DVC** to a Google Drive remote. |
| FR-15 | The system **shall monitor** model performance daily and alert when MAPE exceeds a threshold. |
| FR-16 | The system **shall trigger** automatic retraining on drift detection or on a schedule. |
| FR-17 | The system **shall enforce** role-based access (Business User, Data Scientist, ML Engineer, Admin). |

## 🛡️ Non-Functional Requirements (NFR)

| Category          | ID     | Requirement |
|-------------------|--------|-------------|
| Performance       | NFR-1  | A forecast for one (store, SKU, 30-day) request **shall return in ≤ 2 s** (p95). |
| Performance       | NFR-2  | Batch forecast of 10 000 series **shall complete in ≤ 5 min**. |
| Accuracy          | NFR-3  | Production model **shall maintain MAPE ≤ 10 %** on rolling 30-day evaluation. |
| Scalability       | NFR-4  | The app **shall scale horizontally** behind a load balancer (stateless containers). |
| Availability      | NFR-5  | The web app **shall achieve 99 % uptime** (excluding planned maintenance). |
| Security          | NFR-6  | Authentication via OAuth2 / username+password hashed with bcrypt. |
| Security          | NFR-7  | All traffic **shall be served over HTTPS (TLS 1.2+)**. |
| Maintainability   | NFR-8  | Code coverage of unit tests **≥ 80 %**. |
| Maintainability   | NFR-9  | All code **shall pass** `ruff` + `black` + `mypy` checks in CI. |
| Reproducibility   | NFR-10 | Every model artifact **shall be re-creatable** from a (data hash, code commit, params) triple. |
| Portability       | NFR-11 | The app **shall run** on any host with Docker ≥ 24. |
| Usability         | NFR-12 | A new user **shall generate their first forecast in ≤ 3 clicks**. |
| Observability     | NFR-13 | Centralized logs (structured JSON) + metrics (Prometheus-compatible). |
| Compliance        | NFR-14 | No PII stored; sales data anonymized at ingestion. |

