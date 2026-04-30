# 20. Testing & Validation Plan

## 20.1 Test Pyramid

```
            ▲  E2E (Selenium / Playwright on Streamlit)
           ─┴─
         ─────  Integration (services ⨯ DB ⨯ MLflow ⨯ DVC)
        ───────
       ─────────  Unit (functions, classes, transforms)
```

| Layer        | Coverage target | Tool                                    |
|--------------|-----------------|------------------------------------------|
| Unit         | ≥ 80 %          | `pytest`, `pytest-cov`, `hypothesis`     |
| Integration  | critical paths  | `pytest`, `testcontainers` (Postgres), `mlflow.pyfunc` |
| ML / Data    | every release   | `great-expectations` / `pandera`         |
| E2E UI       | smoke + happy   | `playwright-python`                      |
| Performance  | p95 latency     | `locust`                                 |
| Security     | dependencies    | `pip-audit`, `bandit`                    |

## 20.2 Unit tests (examples)

| Module                        | Test                                    |
|-------------------------------|-----------------------------------------|
| `data.cleaner`                | Removes duplicates, fills NaN per group |
| `data.validator`              | Rejects missing required columns        |
| `features.engineer`           | Lag-1 equals previous row's target       |
| `features.engineer`           | Rolling mean window=7 length matches     |
| `models.xgb_forecaster`       | `fit` then `predict` returns shape (n,)  |
| `models.tuner`                | Best params reduce MAPE vs default       |
| `services.forecast_service`   | Loads production model, returns DataFrame|
| `mlops.drift`                 | PSI ≈ 0 for identical distributions      |

## 20.3 Data validation (Pandera schema)

```python
import pandera as pa
schema = pa.DataFrameSchema({
    "date":       pa.Column(pa.DateTime, nullable=False),
    "store_id":   pa.Column(int, pa.Check.ge(0)),
    "sku_id":     pa.Column(int, pa.Check.ge(0)),
    "units_sold": pa.Column(int, pa.Check.ge(0)),
    "promo":      pa.Column(bool),
    "holiday":    pa.Column(bool),
})
```

Run on every ingest. Failure → reject upload + log error.

## 20.4 Model validation gates (pre-promotion)

A new model can be promoted to **Production** only if **all** gates pass:

| Gate                 | Threshold                          |
|----------------------|------------------------------------|
| Hold-out MAPE        | ≤ 10 %                             |
| RMSE vs prev prod    | ≤ +5 % (no significant regression) |
| Forecast bias        | \|mean(yhat-y)\| / mean(y) < 5 %    |
| Residual stationarity| ADF p-value < 0.05                 |
| Inference latency    | p95 < 200 ms / 30-day forecast     |
| Backtest (12 windows)| MAPE std < 3 %                     |

## 20.5 Integration tests
- Spin up Postgres + MLflow via `docker-compose -f tests/compose.test.yml up`.
- Run `train.py` end-to-end on a fixture dataset, assert model registered.
- Hit `/forecast` API; assert response schema and latency.

## 20.6 E2E (Streamlit)
- Launch app on port 8501 in CI.
- Playwright script:
  1. Login → select store/SKU → click "Generate Forecast".
  2. Assert chart rendered (`canvas` present), download CSV, validate columns.

## 20.7 User Acceptance Testing (UAT)

| ID   | Scenario                                              | Acceptance criterion                              |
|------|-------------------------------------------------------|---------------------------------------------------|
| UAT-1| Upload last quarter's CSV                              | Validation passes; preview shows 100 rows         |
| UAT-2| Generate 30-day forecast for top-selling SKU           | Chart renders in < 2 s; CI band visible            |
| UAT-3| Compare promo vs no-promo scenario                     | Δ revenue and Δ units displayed                    |
| UAT-4| Download forecast CSV                                  | Opens in Excel without errors                     |
| UAT-5| Trigger retraining (admin)                             | Job completes; new model appears in registry      |
| UAT-6| Monitoring shows drift after injecting noisy data      | Alert fires within 24 h                           |

Sign-off: Product Owner + 1 Business User per UAT.

## 20.8 CI workflow (GitHub Actions)
```
lint → unit tests → integration tests → build docker image →
security scan (trivy) → push image → deploy to staging → smoke E2E
```

