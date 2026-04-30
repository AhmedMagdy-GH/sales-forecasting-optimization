# 16. Technology Stack

## 16.1 Summary

| Layer              | Technology                                         | Why |
|--------------------|----------------------------------------------------|-----|
| **Language**       | Python 3.11                                        | ML ecosystem, team familiarity |
| **Frontend / UI**  | **Streamlit 1.36+**, Plotly, streamlit-authenticator | Fast ML UIs, multi-page native support |
| **ML / Modeling**  | **XGBoost 2.x**, scikit-learn, Optuna, Pandas, NumPy | Best-in-class gradient boosting for tabular time-series |
| **Feature Eng.**   | Pandas, Featuretools (optional), holidays           | Lags, rolling stats, calendar, holiday flags |
| **Experiment Mgmt**| **MLflow** (Tracking + Registry)                    | Standard MLOps stack |
| **Data Versioning**| **DVC** with **Google Drive** remote                | Lightweight, already configured in repo |
| **Storage**        | Parquet (local/cloud), SQLite for predictions, PostgreSQL (optional prod) | Fast columnar reads + simple OLTP |
| **Scheduling**     | GitHub Actions / cron (or Airflow) for retraining    | Free for OSS, declarative pipelines |
| **Containerization**| Docker + docker-compose                            | Reproducible deploy |
| **Reverse Proxy**  | Nginx / Caddy (TLS termination)                    | HTTPS, gzip, rate-limit |
| **CI/CD**          | GitHub Actions                                     | Lint, test, build, push image |
| **Monitoring**     | Prometheus + Grafana, structured logs (loguru)      | Drift dashboards, app metrics |
| **Hosting**        | Streamlit Community Cloud / AWS ECS / GCP Cloud Run | Pick per cost & scale |

## 16.2 Repository layout (target)

```
sales-forecasting-optimization/
├── Data/
│   ├── Raw.dvc
│   └── Preprocessed.dvc
├── Docs/
│   └── Diagrams/                ← this folder
├── Notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── validator.py
│   │   └── cleaner.py
│   ├── features/
│   │   └── engineer.py
│   ├── models/
│   │   ├── base.py
│   │   ├── xgb_forecaster.py
│   │   └── tuner.py
│   ├── mlops/
│   │   ├── tracking.py          # MLflow wrapper
│   │   └── drift.py
│   ├── services/
│   │   ├── forecast_service.py
│   │   ├── eda_service.py
│   │   └── upload_service.py
│   ├── app/
│   │   ├── streamlit_app.py
│   │   └── pages/
│   │       ├── 1_📥_Upload.py
│   │       ├── 2_🔎_EDA.py
│   │       ├── 3_📈_Forecast.py
│   │       ├── 4_🧪_Scenarios.py
│   │       └── 5_📊_Monitoring.py
│   └── train.py                 # CLI training entry point
├── tests/
├── conf/
│   └── train.yaml
├── .streamlit/config.toml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 16.3 Key Python dependencies (`requirements.txt`)

```
streamlit>=1.36
xgboost>=2.0
scikit-learn>=1.4
pandas>=2.2
numpy>=1.26
optuna>=3.6
mlflow>=2.13
dvc[gdrive]>=3.50
plotly>=5.22
pyarrow>=15
holidays>=0.50
streamlit-authenticator>=0.3
loguru>=0.7
pydantic>=2.7
pytest>=8.2
ruff>=0.5
black>=24.4
mypy>=1.10
```

