# Sales Forecasting and System Optimization

This project is a sales forecasting system using DVC for data versioning.

## to setup:

- git clone <repo-url>
- pip install "dvc[gdrive]"
- dvc remote modify --local mydrive gdrive_client_id 'YOUR_CLIENT_ID'
- dvc remote modify --local mydrive gdrive_client_secret 'YOUR_CLIENT_SECRET'
- dvc pull   # browser opens → he logs in with his Gmail