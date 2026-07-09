"""
batch_predict.py
----------------
Run the trained model over a CSV of many rows at once (batch inference),
as opposed to `app.py`'s one-row-at-a-time real-time UI. Every row is also
written to the monitoring log so batch traffic is covered by the same
drift/performance monitoring as the Streamlit app.

Usage:
    python src/batch_predict.py --input stores_to_forecast.csv --output predictions.csv

Input CSV must contain the 19 model feature columns (see constants.FEATURES
for the exact names/order). Extra columns are ignored and passed through to
the output unchanged.

Tip: if you'd rather not use the command line at all, the same batch scoring
is available as a point-and-click upload on the "Batch Predictions" page in
the Streamlit app.
"""

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from constants import FEATURES, DEFAULT_MODEL_NAME
from monitoring import PredictionLogger


def main():
    parser = argparse.ArgumentParser(description="Batch sales forecasting")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to write predictions CSV")
    parser.add_argument("--model", default="src/model.pkl", help="Path to trained model pickle")
    parser.add_argument("--no-log", action="store_true",
                        help="Skip writing predictions to the monitoring log "
                             "(src/log/monitoring.csv) — useful for a throwaway test run")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        print(f"ERROR: input CSV is missing required columns: {missing_cols}", file=sys.stderr)
        sys.exit(1)

    model = joblib.load(args.model)

    # Predict on the log scale, then invert the log1p transform used during
    # training (np.expm1 is the exact inverse of np.log1p) to get real €.
    log_scale_predictions = model.predict(df[FEATURES])
    df["PredictedSales"] = np.expm1(log_scale_predictions)

    if not args.no_log:
        logger = PredictionLogger(model_version=DEFAULT_MODEL_NAME)
        for _, row in df.iterrows():
            logger.log(row[FEATURES].to_dict(), float(row["PredictedSales"]))

    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} predictions to {args.output}")
    print(df["PredictedSales"].describe())


if __name__ == "__main__":
    main()
