import pandas as pd
from datetime import datetime

from config import PERFORMANCE_LOG_PATH

# ==========================================
# Model Performance Monitoring
# ==========================================

MODEL_NAME = "XGBoost"
MODEL_VERSION = "v2"


def log_prediction_performance(
    prediction: float,
    execution_time_ms: float,
    status: str = "Success"
) -> None:
    """
    Append a single performance log record to the performance log CSV.

    Stores:
    - Timestamp (datetime.now())
    - Model Name
    - Model Version
    - Prediction
    - Execution Time (ms)
    - Status

    Creates the performance log file automatically if it does not exist,
    and appends new rows to it otherwise, without re-reading the
    existing file contents.
    """
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model Name": MODEL_NAME,
        "Model Version": MODEL_VERSION,
        "Prediction": round(prediction, 2),
        "Execution Time (ms)": round(execution_time_ms, 2),
        "Status": status,
    }

    new_record = pd.DataFrame([record])

    file_exists = PERFORMANCE_LOG_PATH.exists()

    new_record.to_csv(
        PERFORMANCE_LOG_PATH,
        mode="a",
        header=not file_exists,
        index=False,
    )


def load_performance_log() -> pd.DataFrame:

    if PERFORMANCE_LOG_PATH.exists():

        return pd.read_csv(PERFORMANCE_LOG_PATH)

    return pd.DataFrame()