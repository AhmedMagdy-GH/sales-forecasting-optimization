
import pandas as pd
import numpy as np

from config import FEATURES
# ==========================================
# PREDICTION HELPERS
# ==========================================

def build_input_dataframe(inputs: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame from the user's inputs, with columns in
    the exact order the model was trained on.  Column ordering is enforced
    explicitly to guard against future dict-ordering surprises.
    """
    return pd.DataFrame([inputs])[FEATURES]



def run_prediction(model, input_df: pd.DataFrame) -> float:
    """
    Run inference through the trained model and convert
    the prediction back from log scale.
    """
    log_pred = model.predict(input_df)[0]
    return float(np.expm1(log_pred))