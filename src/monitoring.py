"""
monitoring.py
-------------
Lightweight, dependency-free (stdlib + numpy/pandas only) model-monitoring
toolkit for the Rossmann Sales Forecasting model.

Responsibilities
================
1. PredictionLogger  - persists every prediction request/response to a local
   CSV file (`log/monitoring.csv`), so we have a full audit trail with no
   extra infrastructure to run.
2. log_feedback()    - lets a store manager (or a nightly batch job) attach
   the *actual* observed sales to a previously logged prediction, closing
   the feedback loop needed for continuous improvement.
3. compute_drift()   - Population Stability Index (PSI) per feature between
   the training reference distribution (`reference_stats.json`) and a
   window of recent production requests. PSI > 0.25 => significant drift.
4. compute_performance() - rolling MAE / RMSE / MAPE over predictions that
   have received feedback (ground truth), to compare against the training
   metrics baked into `model_meta.json`.

Storage
=======
All runtime monitoring artifacts live under `src/log/`:
    src/log/monitoring.csv  - one row per logged prediction (the audit trail)
    src/log/alerts.log      - one row per raised alert, as JSON lines —
                              the single source of truth for alerts (both
                              written and read by this module)

Plain files are used instead of a database so the audit trail can be opened
directly in Excel/Sheets or read with any text editor, with no dependency on
SQLite or any other database engine. This is intentionally simple and fine
for the traffic volumes this project sees; for high-concurrency production
use, swap the read/write helpers below for a real database — the public
function signatures (log, recent, log_feedback, log_alert, recent_alerts)
wouldn't need to change.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "log")
PREDICTIONS_CSV = os.path.join(LOG_DIR, "monitoring.csv")
ALERTS_LOG_PATH = os.path.join(LOG_DIR, "alerts.log")
REFERENCE_STATS_PATH = os.path.join(BASE_DIR, "reference_stats.json")

PSI_WARNING_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.25

PREDICTION_COLUMNS = [
    "id", "timestamp", "model_version", "inputs_json",
    "prediction", "actual_sales", "feedback_timestamp",
]
ALERT_COLUMNS = ["id", "timestamp", "severity", "source", "message"]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Create log/ and the predictions CSV (with header) if they don't exist yet.

    Important: this also covers the case where the file already *exists* but
    is empty (0 bytes) — e.g. created as a placeholder by a sync tool, or left
    behind after a crash mid-write. Without this check, the very first logged
    prediction would be appended with no header row above it, and every
    future read would then misread that first data row as the column header
    (surfacing as a confusing "KeyError: 'id'" everywhere `id` is used).
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(PREDICTIONS_CSV) or os.path.getsize(PREDICTIONS_CSV) == 0:
        pd.DataFrame(columns=PREDICTION_COLUMNS).to_csv(PREDICTIONS_CSV, index=False)
    if not os.path.exists(ALERTS_LOG_PATH):
        open(ALERTS_LOG_PATH, "a").close()  # touch so it always exists once initialised


PREDICTION_DTYPES = {
    "id": "int64",
    "timestamp": "string",
    "model_version": "string",
    "inputs_json": "string",
    "prediction": "float64",
    "actual_sales": "float64",
    "feedback_timestamp": "string",
}


def _read_predictions() -> pd.DataFrame:
    init_db()
    if os.path.getsize(PREDICTIONS_CSV) == 0:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    # Explicit dtypes matter here: an all-empty column (e.g. feedback_timestamp
    # before any feedback exists) would otherwise be inferred as float64,
    # which then can't hold a timestamp string once feedback is logged.
    df = pd.read_csv(PREDICTIONS_CSV, dtype=PREDICTION_DTYPES, keep_default_na=True)

    if "id" not in df.columns:
        # The file lost its header row at some point (see init_db's docstring
        # for how that can happen) and pandas misread the first data row as
        # the header instead. Recover by re-reading with no header assumed,
        # then rewrite the file with the correct header so this self-heals
        # permanently rather than erroring out every time.
        df = pd.read_csv(PREDICTIONS_CSV, header=None, names=PREDICTION_COLUMNS,
                          dtype=PREDICTION_DTYPES, keep_default_na=True)
        df.to_csv(PREDICTIONS_CSV, index=False)

    return df


def _read_alert_lines() -> list:
    init_db()
    if not os.path.exists(ALERTS_LOG_PATH) or os.path.getsize(ALERTS_LOG_PATH) == 0:
        return []
    rows = []
    with open(ALERTS_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip any malformed/partial line rather than fail the whole read
    return rows


def _sanitize_inputs(inputs: dict) -> dict:
    """
    Normalize a raw feature dict before it's persisted, so values are never
    silently upcast to floats (a common pandas gotcha when a mixed-dtype
    DataFrame row is converted via `row.to_dict()`, which forces every value
    in that row to a single common dtype). Without this, a categorical value
    like Promo=1 can be logged as 1.0, which then fails to match the "1"
    string key in reference_stats.json and produces bogus drift alerts.
    """
    clean = {}
    for k, v in inputs.items():
        if isinstance(v, (bool, np.bool_)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            fv = float(v)
            clean[k] = int(fv) if fv.is_integer() else fv
        elif isinstance(v, (np.integer,)):
            clean[k] = int(v)
        else:
            clean[k] = v
    return clean


class PredictionLogger:
    """Append-only log of every inference request/response, backed by a CSV file."""

    def __init__(self, model_version: str = "XGBoost_tuned"):
        init_db()
        self.model_version = model_version

    def log(self, inputs: dict, prediction: float) -> int:
        clean_inputs = _sanitize_inputs(inputs)
        ts = datetime.now(timezone.utc).isoformat()

        existing = _read_predictions()
        new_id = 1 if existing.empty else int(existing["id"].max()) + 1

        row = pd.DataFrame([{
            "id": new_id,
            "timestamp": ts,
            "model_version": self.model_version,
            "inputs_json": json.dumps(clean_inputs),
            "prediction": float(prediction),
            "actual_sales": np.nan,
            "feedback_timestamp": "",
        }])
        row.to_csv(PREDICTIONS_CSV, mode="a", header=False, index=False)
        return new_id

    def recent(self, limit: int = 500) -> pd.DataFrame:
        df = _read_predictions()
        if df.empty:
            return df
        return df.sort_values("id", ascending=False).head(limit).reset_index(drop=True)

    def count(self) -> int:
        return len(_read_predictions())


def log_feedback(prediction_id: int, actual_sales: float) -> bool:
    """
    Attach ground-truth sales to a previously logged prediction.
    Returns True if a matching prediction_id was found and updated,
    False if no row with that ID exists (e.g. a typo'd ID).
    """
    df = _read_predictions()
    mask = df["id"] == prediction_id
    if not mask.any():
        return False
    ts = datetime.now(timezone.utc).isoformat()
    df.loc[mask, "actual_sales"] = float(actual_sales)
    df.loc[mask, "feedback_timestamp"] = ts
    df.to_csv(PREDICTIONS_CSV, index=False)
    return True


def log_alert(severity: str, source: str, message: str) -> None:
    """
    Append one alert to the single alerts.log file (JSON lines), which is
    the sole source of truth for alerts — no separate database/CSV table.
    """
    existing = _read_alert_lines()
    new_id = 1 if not existing else max(r.get("id", 0) for r in existing) + 1
    ts = datetime.now(timezone.utc).isoformat()
    entry = {"id": new_id, "timestamp": ts, "severity": severity, "source": source, "message": message}
    with open(ALERTS_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def recent_alerts(limit: int = 100) -> pd.DataFrame:
    rows = _read_alert_lines()
    if not rows:
        return pd.DataFrame(columns=ALERT_COLUMNS)
    df = pd.DataFrame(rows)
    for col in ALERT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[ALERT_COLUMNS].sort_values("id", ascending=False).head(limit).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Drift detection (Population Stability Index)
# ---------------------------------------------------------------------------

def _load_reference_stats() -> dict:
    with open(REFERENCE_STATS_PATH) as f:
        return json.load(f)


def _psi_numeric(recent_values: pd.Series, ref: dict) -> float:
    edges = np.array(ref["bin_edges"])
    ref_props = np.array(ref["proportions"])
    counts, _ = np.histogram(recent_values.dropna(), bins=edges)
    total = counts.sum()
    if total == 0:
        return 0.0
    cur_props = counts / total
    eps = 1e-4
    ref_props = np.clip(ref_props, eps, None)
    cur_props = np.clip(cur_props, eps, None)
    return float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))


def _normalize_cat_key(v) -> str:
    """
    Canonicalize a categorical value to a comparable string, regardless of
    whether it arrived as an int, a float (1.0), a bool, or a stringified
    version of any of those ("1", "1.0", "True"). This is what lets PSI
    compare current traffic against the reference distribution correctly
    even when older logged rows used a different (buggy) representation.
    """
    if isinstance(v, str):
        if v in ("True", "False"):
            return "1" if v == "True" else "0"
        try:
            f = float(v)
        except ValueError:
            return v
    elif isinstance(v, bool):
        return "1" if v else "0"
    else:
        f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


def _merge_normalized(dist: dict) -> dict:
    merged: dict = {}
    for k, v in dist.items():
        nk = _normalize_cat_key(k)
        merged[nk] = merged.get(nk, 0.0) + v
    return merged


def _psi_categorical(recent_values: pd.Series, ref: dict) -> float:
    ref_dist = _merge_normalized(ref["distribution"])
    cur_raw = recent_values.value_counts(normalize=True).to_dict()
    cur_dist = _merge_normalized(cur_raw)
    categories = set(ref_dist) | set(cur_dist)
    eps = 1e-4
    psi = 0.0
    for cat in categories:
        r = max(ref_dist.get(cat, 0.0), eps)
        c = max(cur_dist.get(cat, 0.0), eps)
        psi += (c - r) * np.log(c / r)
    return float(psi)


def compute_drift(recent_df: Optional[pd.DataFrame] = None, window: int = 200) -> pd.DataFrame:
    """
    Returns a DataFrame: feature | psi | status ('ok' | 'warning' | 'critical')
    comparing the last `window` production requests against the training
    reference distribution.
    """
    ref_stats = _load_reference_stats()

    if recent_df is None:
        logger = PredictionLogger()
        recent_df = logger.recent(limit=window)

    if recent_df.empty:
        return pd.DataFrame(columns=["feature", "psi", "status"])

    inputs = pd.json_normalize(recent_df["inputs_json"].apply(json.loads))

    rows = []
    for feature, ref in ref_stats.items():
        if feature not in inputs.columns:
            continue
        series = inputs[feature]
        if ref["type"] == "numeric":
            psi = _psi_numeric(series, ref)
        else:
            psi = _psi_categorical(series, ref)

        if psi >= PSI_CRITICAL_THRESHOLD:
            status = "critical"
        elif psi >= PSI_WARNING_THRESHOLD:
            status = "warning"
        else:
            status = "ok"
        rows.append({"feature": feature, "psi": round(psi, 4), "status": status})

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Performance monitoring (needs ground-truth feedback)
# ---------------------------------------------------------------------------

def compute_performance(window: int = 500) -> Optional[dict]:
    """
    Rolling MAE / RMSE / MAPE over predictions that have received feedback.
    Returns None if no feedback has been recorded yet.
    """
    logger = PredictionLogger()
    df = logger.recent(limit=window)
    df = df.dropna(subset=["actual_sales"])
    if df.empty:
        return None

    errors = df["prediction"] - df["actual_sales"]
    mae = errors.abs().mean()
    rmse = float(np.sqrt((errors ** 2).mean()))
    nonzero = df["actual_sales"].replace(0, np.nan)
    mape = float((errors.abs() / nonzero).dropna().mean())

    return {
        "n_samples": int(len(df)),
        "mae": float(mae),
        "rmse": rmse,
        "mape": mape,
    }


if __name__ == "__main__":
    init_db()
    print(f"Monitoring storage initialised at {LOG_DIR}")
