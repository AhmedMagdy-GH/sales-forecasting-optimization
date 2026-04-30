# Sales Forecasting and System Optimization

This project is a sales forecasting system using DVC for data versioning.

## to setup:

- git clone <repo-url>
- pip install "dvc[gdrive]"
- dvc remote modify --local mydrive gdrive_client_id 'YOUR_CLIENT_ID'
- dvc remote modify --local mydrive gdrive_client_secret 'YOUR_CLIENT_SECRET'
- dvc pull   # browser opens → he logs in with his Gmail
```


DEPI_Project
├─ .dvc
│  ├─ cache
│  ├─ config
│  ├─ config.local
│  └─ tmp
├─ .dvcignore
├─ Data
│  ├─ Preprocessed
│  ├─ Preprocessed.dvc
│  ├─ Raw
│  └─ Raw.dvc
├─ Docs
│  ├─ DEPI-docs
│  │  ├─ IBM Data Science Project - Round4.pdf
│  │  ├─ Project Documentation.pdf
│  │  └─ Project-2.pdf
│  └─ DVC-infos
│     ├─ DVC-commands.yaml
│     ├─ DVC-Info.yaml
│     └─ dvc_gdrive_setup.yaml
├─ Notebooks
│  └─ test.py
├─ README.md
└─ src
   └─ test.py

```