# 21. Deployment Strategy

## 21.1 Environments

| Env        | Purpose                                | URL                                | Branch  |
|------------|----------------------------------------|------------------------------------|---------|
| `local`    | Developer machine, hot-reload Streamlit| `http://localhost:8501`            | any     |
| `staging`  | Pre-prod, smoke tests + UAT            | `https://staging.sales-forecast...`| `develop` |
| `production`| Live for business users                | `https://sales-forecast...`        | `main`  |

## 21.2 Hosting Options (pick one)

| Option                       | Pros                                  | Cons                          |
|------------------------------|---------------------------------------|-------------------------------|
| **Streamlit Community Cloud**| Free, zero ops, GitHub-native         | 1 GB RAM, public only         |
| **AWS ECS Fargate + ALB**    | Auto-scale, VPC, ACM TLS              | Higher cost, more setup       |
| **GCP Cloud Run**            | Pay-per-request, fast cold starts     | 1 container = 1 user (sticky) |
| **Self-host VM + Docker**    | Cheapest at scale, full control       | You own the SRE work          |

We recommend **GCP Cloud Run** (or AWS ECS) for production.

## 21.3 Containerization

`Dockerfile` (multi-stage):
```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ src/
COPY .streamlit/ .streamlit/
EXPOSE 8501
HEALTHCHECK CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "src/app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
```

`docker-compose.yml` (local stack):
```yaml
services:
  app:        { build: ., ports: ["8501:8501"], depends_on: [mlflow, db] }
  mlflow:     { image: ghcr.io/mlflow/mlflow:v2.13.0, ports: ["5000:5000"],
                command: mlflow server --host 0.0.0.0
                  --backend-store-uri postgresql://mlflow:mlflow@db/mlflow
                  --artifacts-destination /mlruns,
                volumes: [mlruns:/mlruns] }
  db:         { image: postgres:16, environment: { POSTGRES_PASSWORD: mlflow } }
volumes: { mlruns: {} }
```

## 21.4 CI/CD Pipeline (GitHub Actions)

```
┌─────────┐  push   ┌──────────┐   pass   ┌──────────┐    build & push    ┌──────────┐
│ developer├────────►│  lint    ├─────────►│  tests   ├───────────────────►│  GHCR /  │
└─────────┘  PR     │ (ruff,   │          │ (pytest, │   docker buildx    │  ECR     │
                    │  black,  │          │  cov,    │                    └─────┬────┘
                    │  mypy)   │          │  e2e)    │                          │
                    └──────────┘          └──────────┘                          │
                                                                                ▼
                                                              ┌──────────────────────────┐
                                                              │  deploy to STAGING       │
                                                              │  (Cloud Run / ECS)       │
                                                              └─────┬────────────────────┘
                                                                    │ smoke + UAT pass
                                                                    ▼
                                                              ┌──────────────────────────┐
                                                              │  deploy to PROD          │
                                                              │  (blue-green / canary)   │
                                                              └──────────────────────────┘
```

Stages:
1. **Build & Test** — lint, unit, integration on every PR.
2. **Image Build** — multi-arch `linux/amd64` & `linux/arm64`, tagged with git SHA + semver.
3. **Vulnerability Scan** — `trivy image`, fail on CRITICAL.
4. **Deploy Staging** — auto on merge to `develop`.
5. **Promote to Prod** — manual approval (release manager) or tag push `v*.*.*`.

## 21.5 Release Strategy
- **Blue/Green** in Cloud Run (revisions) or ECS (target groups).
- 10 % canary traffic for 30 min, auto-rollback if `/_stcore/health` fails or
  Prometheus error-rate > 1 %.

## 21.6 Scaling

| Resource              | Strategy                                |
|-----------------------|-----------------------------------------|
| Streamlit containers  | HPA on CPU > 60 % or concurrency > 40   |
| MLflow server         | Single instance + RDS (vertical scale)  |
| Training jobs         | Spot VMs / Cloud Run Jobs (on demand)   |
| Postgres              | RDS Multi-AZ; read replica for analytics|
| Object storage        | S3 / GCS — virtually unlimited          |

## 21.7 Configuration & Secrets
- 12-factor: all config via env vars.
- Secrets in **AWS Secrets Manager** / **GCP Secret Manager**.
- `.env.example` checked in; real `.env` ignored by git.

## 21.8 Backup & DR
- Postgres: daily automated snapshots, 30-day retention, PITR.
- DVC remote (GDrive) + mirror to S3 weekly.
- MLflow artifacts stored in versioned S3 bucket.
- RPO 24 h, RTO 4 h.

## 21.9 Observability
- **Logs**: structured JSON via `loguru`, shipped to Loki / CloudWatch.
- **Metrics**: `/metrics` Prometheus endpoint (request count, latency, model MAPE).
- **Tracing**: optional OpenTelemetry → Tempo / X-Ray.
- **Alerts** (Grafana / PagerDuty):
  - 5xx rate > 1 % for 5 min
  - p95 latency > 2 s for 10 min
  - Rolling MAPE > 10 % for 24 h
  - Daily training job failure

