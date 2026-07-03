# 📊 Sales Forecasting & System Optimization — Rossmann Store Sales

An end-to-end machine learning project that predicts daily sales for Rossmann drug stores and surfaces insights for business optimization — from raw transaction data all the way to a deployed Streamlit app, with full experiment tracking (MLflow) and data versioning (DVC).

---

## 🚀 Project Overview

This project builds a robust pipeline for:

* 📈 **Sales Forecasting** using historical Rossmann transaction data
* ⚙️ **System Optimization** to support inventory and staffing decisions
* 🧹 **Data Cleaning & Feature Engineering**
* 📊 **Exploratory Data Analysis (EDA)** to uncover sales drivers
* 🤖 **Machine Learning Modeling** with LightGBM, tracked via MLflow
* 🌐 **Deployment** as an interactive Streamlit web app

The goal is to help store managers make smarter decisions, reduce demand uncertainty, and maximize revenue.

---

## 🧠 Problem Statement

Rossmann store managers are asked to predict daily sales up to six weeks in advance. Sales are influenced by many factors — promotions, competition, school and state holidays, seasonality, and locality — that are hard to account for manually. This project builds a predictive system that:

* Forecasts future daily sales per store
* Identifies the key drivers of sales performance
* Supports optimization decisions around inventory and staffing

---

## 🗂️ Project Structure

```
DEPI_Project/
├── .dvc/                          # DVC internals (cache, config) — data versioning
├── .dvcignore
├── .vscode/                       # Editor settings
├── Data/
│   ├── Raw/                       # historical_sales.csv, store_details.csv (DVC-tracked)
│   ├── Raw.dvc
│   ├── Preprocessed/              # cleaned & feature-engineered datasets (DVC-tracked)
│   │   ├── cleaned_training_data_V1.parquet
│   │   ├── featured_training_data_V2.parquet
│   │   └── merged_training_data.csv
│   └── Preprocessed.dvc
├── Docs/
│   ├── DEPI-docx/                 # IBM DEPI submission docs (PDF)
│   ├── DVC-infos/                 # DVC command reference & Google Drive remote setup
│   └── Required-docx/             # Literature Review, Project Plan, Proposal, System Analysis & Design
├── Notebooks/
│   ├── 01_data_exploring.ipynb
│   ├── 02_data_ingestion.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_Modeling.ipynb          # LightGBM training + MLflow logging
│   └── mlruns/                    # local MLflow tracking artifacts
├── mlruns/                        # MLflow experiment tracking store (models, metrics, params)
├── src/
│   └── app.py                     # Streamlit deployment app
├── requirements.txt
└── README.md
```

---

## 🔄 Workflow

### 1. Data Ingestion
* `historical_sales.csv` — daily sales transactions per store
* `store_details.csv` — store metadata (type, assortment, competition distance, promo intervals)
* Raw data versioned and pulled via **DVC** with a Google Drive remote

### 2. Data Cleaning
* Handling missing values (competition/promo fields, etc.)
* Removing outliers
* Data type optimization

### 3. Feature Engineering
* Time-based features (day, week, month, year)
* Lag and rolling sales features
* Promo, holiday, and competition-based features
* Encoding categorical variables
* Output saved as versioned Parquet datasets (`cleaned_training_data_V1`, `featured_training_data_V2`)

### 4. Exploratory Data Analysis (EDA)
* Sales trends over time
* Store type / assortment performance comparison
* Impact of promotions and holidays on sales

### 5. Model Training
* **LightGBM** (primary model), built with **scikit-learn Pipelines**
* Experiment tracking, parameter logging, and model registry via **MLflow**
* Registered model: `rossmann-sales-forecasting-model`

### 6. Evaluation
* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score
* Metrics logged per run in MLflow (including original-scale MAE/RMSE)

### 7. Deployment
* Interactive **Streamlit** app (`src/app.py`) serving the trained model for live sales predictions

---

## 📊 Key Insights

* Sales are highly influenced by promotions
* Seasonal trends significantly affect revenue
* Store type and location play a critical role
* Outlier removal improves model performance

---

## ⚙️ System Optimization Module

This part focuses on improving business operations by:

* 📦 Optimizing inventory levels
* 🕒 Improving staffing schedules
* 💰 Reducing operational costs
* 📍 Identifying underperforming stores

---

## 🧪 Tech Stack

* Python 3.11 🐍
* Pandas & NumPy
* PyArrow / fastparquet
* Scikit-learn (Pipelines)
* LightGBM / XGBoost
* Matplotlib & Seaborn
* MLflow (experiment tracking & model registry)
* DVC (data version control, Google Drive remote)
* Streamlit (deployment)
* Jupyter Notebook

---

## 📌 Future Improvements

* Deploy via FastAPI as a REST API alongside the Streamlit app
* Add real-time data streaming
* Integrate deep learning models (LSTM) for longer-horizon forecasts
* Expand system optimization module with inventory simulation

---

## ⚙️ Setup

Data is stored on Google Drive and pulled via DVC using a **shared Service Account** (no per-user browser login, tokens never expire — see `Docs/DVC-infos/dvc_gdrive_setup.yaml` for the full walkthrough).

```bash
# 1. Clone the repo
git clone <repo-url>
cd DEPI_Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get the service account key file
#    Ask a project maintainer to privately share the JSON key
#    (e.g. via WhatsApp/email — never through Git).
#    Save it anywhere on your machine, e.g. ./gdrive-service-account.json

# 4. Point DVC at the key (path is stored locally in .dvc/config.local, gitignored)
dvc remote modify mydrive gdrive_use_service_account true
dvc remote modify mydrive --local gdrive_service_account_json_file_path "/path/to/gdrive-service-account.json"

# 5. Pull data & model artifacts — authenticates silently, no browser needed
dvc pull

# 6. Reproduce the pipeline (EDA → cleaning → features → modeling)
jupyter notebook Notebooks/

# 7. Run the deployed app locally
streamlit run src/app.py
```

> ⚠️ **Note:** Never commit the service-account JSON key, `Credentials.json`, or `.dvc/config.local` to Git. Make sure they're all listed in `.gitignore`. Each teammate only runs step 4 once — after that, `dvc pull`/`dvc push` work indefinitely with no re-authentication.

---

## 📈 Example Results

| Metric | Value |
| ------ | ----- |
| MAE    | 0.12  |
| RMSE   | 0.18  |
| R²     | 0.91  |
