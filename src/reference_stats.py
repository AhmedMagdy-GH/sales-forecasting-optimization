"""
reference_stats.py
-------------------
Builds a reference-distribution snapshot of the training data so that
incoming production traffic can later be compared against it for drift
detection (see monitoring.py -> compute_drift()).

Run once after every retraining:
    python src/reference_stats.py

Output:
    src/reference_stats.json

Notes on data availability
---------------------------
The fully feature-engineered training table used for the final model
(`featured_training_data_V2.parquet`, referenced in README.md) is DVC-tracked
but was not present in the delivered archive. Where a raw source column maps
1:1 to a model feature (DayOfWeek, CompetitionDistance, StateHoliday,
SchoolHoliday, Promo, Promo2, StoreType, Assortment) stats are computed
directly from `Data/Preprocessed/cleaned_training_data_V1.parquet`.

For engineered features that only exist after feature engineering
(Store_TargetEnc, Week, Year, IsWeekend, IsMonthStart, IsMonthEnd,
CompetitionOpenMonths, Promo2ActiveWeeks, IsPromo2Active,
CompetitionDistanceMissing, CompetitionOpenMissing) we fall back to the
training-distribution ranges already defined in `src/constants.py`
(`FEATURE_RANGES`) plus reasonable assumed distributions. These are clearly
flagged with `"source": "assumed"` in the output JSON so anyone re-running
this script with the real V2 dataset can see exactly what to replace.
"""

import json
import os
import numpy as np
import pandas as pd

from constants import FEATURES

NUMERIC_BIN_FEATURES = [
    'Store_TargetEnc', 'CompetitionDistance', 'CompetitionOpenMonths',
    'Promo2ActiveWeeks', 'Week', 'Year',
]

CATEGORICAL_FEATURES = [
    'DayOfWeek', 'CompetitionDistanceMissing', 'CompetitionOpenMissing',
    'StateHoliday', 'SchoolHoliday', 'Promo', 'Promo2', 'StoreType',
    'Assortment', 'IsWeekend', 'IsMonthStart', 'IsMonthEnd', 'IsPromo2Active',
]

N_BINS = 10


def _quantile_bins(series: pd.Series, n_bins: int = N_BINS):
    series = series.dropna()
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(series, quantiles))
    if len(edges) < 2:
        edges = np.array([series.min() - 1, series.max() + 1])
    counts, edges = np.histogram(series, bins=edges)
    proportions = (counts / counts.sum()).tolist()
    return {"bin_edges": edges.tolist(), "proportions": proportions,
            "mean": float(series.mean()), "std": float(series.std()),
            "min": float(series.min()), "max": float(series.max())}


def _categorical_dist(series: pd.Series):
    vc = series.value_counts(normalize=True)
    return {str(k): float(v) for k, v in vc.items()}


def build_from_raw(raw_path: str) -> dict:
    df = pd.read_parquet(raw_path)

    label_maps = {
        "StoreType": {"a": 0, "b": 1, "c": 2, "d": 3},
        "Assortment": {"a": 0, "b": 1, "c": 2},
        "StateHoliday": {"0": 0, "a": 1, "b": 1, "c": 1},
    }
    for col, mapping in label_maps.items():
        if col in df.columns and df[col].dtype.name in ("category", "object", "string"):
            df[col] = df[col].astype(str).map(mapping).fillna(0).astype(int)
    for col in ("Promo", "Promo2", "CompetitionDistanceMissing", "CompetitionOpenMissing"):
        if col in df.columns:
            df[col] = df[col].astype(int)

    stats = {}

    for col in NUMERIC_BIN_FEATURES:
        if col in df.columns:
            stats[col] = {"type": "numeric", "source": "training_data",
                          **_quantile_bins(df[col])}

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            stats[col] = {"type": "categorical", "source": "training_data",
                          "distribution": _categorical_dist(df[col])}

    return stats


# Assumed distributions for engineered features not present in the raw
# preprocessed table shipped with this project. Replace by re-running this
# script against `featured_training_data_V2.parquet` once available.
ASSUMED_STATS = {
    "Store_TargetEnc": {
        "type": "numeric", "source": "assumed",
        "mean": 8.5, "std": 1.1, "min": 5.0, "max": 12.0,
        "bin_edges": list(np.linspace(5.0, 12.0, N_BINS + 1)),
        "proportions": [1.0 / N_BINS] * N_BINS,
    },
    "CompetitionOpenMonths": {
        "type": "numeric", "source": "assumed",
        "mean": 60.0, "std": 45.0, "min": 0.0, "max": 600.0,
        "bin_edges": list(np.linspace(0.0, 600.0, N_BINS + 1)),
        "proportions": [1.0 / N_BINS] * N_BINS,
    },
    "Promo2ActiveWeeks": {
        "type": "numeric", "source": "assumed",
        "mean": 40.0, "std": 35.0, "min": 0.0, "max": 300.0,
        "bin_edges": list(np.linspace(0.0, 300.0, N_BINS + 1)),
        "proportions": [1.0 / N_BINS] * N_BINS,
    },
    "Week": {
        "type": "numeric", "source": "assumed",
        "mean": 26.5, "std": 15.0, "min": 1, "max": 53,
        "bin_edges": list(np.linspace(1, 53, N_BINS + 1)),
        "proportions": [1.0 / N_BINS] * N_BINS,
    },
    "Year": {
        "type": "numeric", "source": "assumed",
        "mean": 2014.5, "std": 0.7, "min": 2013, "max": 2015,
        "bin_edges": list(np.linspace(2013, 2015, N_BINS + 1)),
        "proportions": [1.0 / N_BINS] * N_BINS,
    },
    "CompetitionDistanceMissing": {"type": "categorical", "source": "assumed",
                                    "distribution": {"0": 0.93, "1": 0.07}},
    "CompetitionOpenMissing": {"type": "categorical", "source": "assumed",
                               "distribution": {"0": 0.68, "1": 0.32}},
    "IsWeekend": {"type": "categorical", "source": "assumed",
                  "distribution": {"0": 0.715, "1": 0.285}},
    "IsMonthStart": {"type": "categorical", "source": "assumed",
                     "distribution": {"0": 0.967, "1": 0.033}},
    "IsMonthEnd": {"type": "categorical", "source": "assumed",
                   "distribution": {"0": 0.967, "1": 0.033}},
    "IsPromo2Active": {"type": "categorical", "source": "assumed",
                       "distribution": {"0": 0.6, "1": 0.4}},
}


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(root, "Data", "Preprocessed", "cleaned_training_data_V1.parquet")

    stats = {}
    if os.path.exists(raw_path):
        stats.update(build_from_raw(raw_path))
    else:
        print(f"WARNING: {raw_path} not found, using only assumed stats.")

    for k, v in ASSUMED_STATS.items():
        if k not in stats:
            stats[k] = v

    missing = [f for f in FEATURES if f not in stats]
    if missing:
        print(f"WARNING: no reference stats produced for: {missing}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote reference stats for {len(stats)} features to {out_path}")


if __name__ == "__main__":
    main()
