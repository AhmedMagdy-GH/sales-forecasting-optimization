import pandas as pd
from datetime import datetime

from config import HISTORY_PATH
from components.result import get_sales_status

# ==========================================
# Prediction Log
# ==========================================


def save_prediction(inputs: dict, prediction: float) -> None:
    """
    Save a single successful prediction to the history CSV.

    Stores:
    - Timestamp (datetime.now())
    - Every model feature/input used for the prediction
    - The predicted sales value
    - The derived sales status/category

    Creates the history file automatically if it does not exist,
    and appends new rows to it otherwise, without re-reading the
    existing file contents.
    """
    sales_status, _ = get_sales_status(prediction)

    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **inputs,
        "Predicted Sales (€)": round(prediction, 2),
        "Sales Level": sales_status,
    }

    new_record = pd.DataFrame([record])

    file_exists = HISTORY_PATH.exists()

    new_record.to_csv(
        HISTORY_PATH,
        mode="a",
        header=not file_exists,
        index=False,
    )


def load_history() -> pd.DataFrame:

    if HISTORY_PATH.exists():

        return pd.read_csv(HISTORY_PATH)

    return pd.DataFrame()