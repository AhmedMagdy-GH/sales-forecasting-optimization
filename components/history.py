import os
import pandas as pd
from datetime import datetime
# ==========================================
# Prediction Log
# ==========================================


HISTORY_FILE = "prediction_log.csv"

def save_prediction(
    store,
    year,
    week,
    promo,
    prediction,
    category,
):

    new_record = pd.DataFrame([{
    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Store": round(store, 2),
    "Year": year,
    "Week": week,
    "Promo": "Yes" if promo else "No",
    "Predicted Sales (€)": round(prediction, 2),
    "Sales Level": category
    }])
    if os.path.exists(HISTORY_FILE):

        history = pd.read_csv(HISTORY_FILE)

        history = pd.concat([history, new_record], ignore_index=True)

    else:

        history = new_record

    history.to_csv(HISTORY_FILE, index=False)


def load_history():

    if os.path.exists(HISTORY_FILE):

        return pd.read_csv(HISTORY_FILE)

    return pd.DataFrame()


