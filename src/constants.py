"""
constants.py
------------
Shared, single-source-of-truth constants for the Rossmann Sales Forecasting
project. Previously the 19-feature list was copy-pasted separately into
app.py, batch_predict.py, reference_stats.py, and pages/2_Batch_Predictions.py
— which meant a future change to the feature set had to be made in four
places and could easily drift out of sync. Importing FEATURES from here
instead keeps every part of the app guaranteed to agree on the exact same
column names and order the model was trained on.
"""

# The 19 engineered features the model expects, in the exact order it was
# trained on. Every prediction (real-time or batch) must supply these columns
# in this order — see docx report "Model Monitoring Setup" §3 for the full
# feature dictionary.
FEATURES = [
    "Store_TargetEnc",
    "DayOfWeek",
    "CompetitionDistance",
    "CompetitionDistanceMissing",
    "CompetitionOpenMissing",
    "StateHoliday",
    "SchoolHoliday",
    "Promo",
    "Promo2",
    "StoreType",
    "Assortment",
    "Year",
    "Week",
    "IsWeekend",
    "IsMonthStart",
    "IsMonthEnd",
    "CompetitionOpenMonths",
    "Promo2ActiveWeeks",
    "IsPromo2Active",
]

# Reasonable input ranges seen during training, used by app.py to warn the
# user (not to block them) when a value looks out of distribution.
FEATURE_RANGES = {
    "Store_TargetEnc": (5.0, 12.0),
    "CompetitionDistance": (0.0, 100_000.0),
    "CompetitionOpenMonths": (0.0, 600.0),
    "Promo2ActiveWeeks": (0.0, 300.0),
    "Week": (1, 53),
    "Year": (2013, 2030),
}

# Default model identity shown in the UI and used to tag logged predictions
# when no model_meta.json is present.
DEFAULT_MODEL_NAME = "XGBoost_tuned"
