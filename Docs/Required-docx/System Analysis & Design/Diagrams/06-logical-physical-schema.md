# 6. Logical & Physical Schema

## 6.1 Logical schema (3NF)

The data model is normalized to **3rd Normal Form**:

- Dimensions: `STORE`, `PRODUCT`, `CATEGORY`, `DATE_DIM`, `PROMOTION`, `USER`
- Facts:      `SALES_FACT`, `WEATHER`, `FORECAST`
- Meta:       `MODEL_RUN`, `AUDIT_LOG`

Star-schema-style fact (`SALES_FACT`) referencing dimension surrogate keys.
`PRODUCT` → `CATEGORY` is many-to-one; `CATEGORY` self-references for hierarchy.

## 6.2 Physical schema (PostgreSQL DDL)

```sql
-- ===== Dimensions =====
CREATE TABLE category (
    category_id   SERIAL PRIMARY KEY,
    name          VARCHAR(80)  NOT NULL,
    parent_id     INT REFERENCES category(category_id)
);

CREATE TABLE product (
    sku_id        SERIAL PRIMARY KEY,
    sku_code      VARCHAR(40)  UNIQUE NOT NULL,
    product_name  VARCHAR(150) NOT NULL,
    category_id   INT NOT NULL REFERENCES category(category_id),
    unit_price    NUMERIC(10,2) CHECK (unit_price >= 0),
    unit_cost     NUMERIC(10,2) CHECK (unit_cost  >= 0)
);

CREATE TABLE store (
    store_id      SERIAL PRIMARY KEY,
    store_name    VARCHAR(100) NOT NULL,
    city          VARCHAR(80),
    region        VARCHAR(80),
    country       VARCHAR(60),
    store_type    VARCHAR(30),
    open_date     DATE
);

CREATE TABLE date_dim (
    date_id       DATE PRIMARY KEY,
    day           SMALLINT,
    week          SMALLINT,
    month         SMALLINT,
    quarter       SMALLINT,
    year          SMALLINT,
    day_of_week   SMALLINT,
    is_weekend    BOOLEAN,
    is_holiday    BOOLEAN,
    holiday_name  VARCHAR(80)
);

CREATE TABLE promotion (
    promo_id      SERIAL PRIMARY KEY,
    name          VARCHAR(100),
    promo_type    VARCHAR(30),
    discount_pct  NUMERIC(5,2) CHECK (discount_pct BETWEEN 0 AND 100),
    start_date    DATE,
    end_date      DATE,
    CHECK (end_date >= start_date)
);

-- ===== Facts =====
CREATE TABLE sales_fact (
    sale_id       BIGSERIAL PRIMARY KEY,
    date_id       DATE NOT NULL REFERENCES date_dim(date_id),
    store_id      INT  NOT NULL REFERENCES store(store_id),
    sku_id        INT  NOT NULL REFERENCES product(sku_id),
    promo_id      INT  REFERENCES promotion(promo_id),
    units_sold    INT  NOT NULL CHECK (units_sold >= 0),
    revenue       NUMERIC(12,2) NOT NULL,
    on_promo      BOOLEAN DEFAULT FALSE
);
CREATE INDEX idx_sales_store_sku_date ON sales_fact(store_id, sku_id, date_id);
CREATE INDEX idx_sales_date           ON sales_fact(date_id);

CREATE TABLE weather (
    weather_id    BIGSERIAL PRIMARY KEY,
    date_id       DATE NOT NULL REFERENCES date_dim(date_id),
    store_id      INT  NOT NULL REFERENCES store(store_id),
    temp_c        NUMERIC(4,1),
    precipitation NUMERIC(5,2),
    condition     VARCHAR(40),
    UNIQUE (date_id, store_id)
);

CREATE TABLE model_run (
    run_id        VARCHAR(64) PRIMARY KEY,        -- MLflow run uuid
    model_name    VARCHAR(80) NOT NULL,
    version       VARCHAR(20),
    algorithm     VARCHAR(40) DEFAULT 'XGBoost',
    params_json   JSONB,
    rmse          NUMERIC(10,4),
    mae           NUMERIC(10,4),
    mape          NUMERIC(6,3),
    trained_at    TIMESTAMPTZ DEFAULT now(),
    status        VARCHAR(20) CHECK (status IN ('staging','production','archived'))
);

CREATE TABLE forecast (
    forecast_id   BIGSERIAL PRIMARY KEY,
    run_id        VARCHAR(64) NOT NULL REFERENCES model_run(run_id),
    store_id      INT  NOT NULL REFERENCES store(store_id),
    sku_id        INT  NOT NULL REFERENCES product(sku_id),
    forecast_date DATE NOT NULL,
    horizon_days  INT  NOT NULL,
    yhat          NUMERIC(12,2),
    yhat_lower    NUMERIC(12,2),
    yhat_upper    NUMERIC(12,2),
    created_at    TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_forecast_lookup ON forecast(store_id, sku_id, forecast_date);

-- ===== Auth & audit =====
CREATE TABLE app_user (
    user_id       SERIAL PRIMARY KEY,
    username      VARCHAR(50)  UNIQUE NOT NULL,
    email         VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role          VARCHAR(30)  CHECK (role IN
                       ('business_user','data_scientist','ml_engineer','admin')),
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE audit_log (
    log_id        BIGSERIAL PRIMARY KEY,
    user_id       INT REFERENCES app_user(user_id),
    action        VARCHAR(80),
    payload       JSONB,
    ts            TIMESTAMPTZ DEFAULT now()
);
```

## 6.3 Normalization notes

| Form  | Status | Justification |
|-------|--------|---------------|
| 1NF   | ✅     | All attributes atomic; no repeating groups. |
| 2NF   | ✅     | All non-key attributes depend on the whole PK (surrogate keys used in facts). |
| 3NF   | ✅     | No transitive dependencies — `category_name` lives only in `category`, not in `product`. |
| BCNF  | ✅     | Every determinant is a candidate key. |

For analytical workloads we **denormalize on read** by materializing a wide
`sales_enriched` view joining `sales_fact ⨝ date_dim ⨝ product ⨝ store ⨝ weather`
which is the input to feature engineering.

