from pathlib import Path

# ==========================================
# PAGE CONFIG
# ==========================================

PAGE_TITLE = "Rossmann Sales Forecasting"
PAGE_ICON = "🏪"
LAYOUT = "wide"

# ==========================================
# FILE PATHS
# ==========================================

MODEL_PATH = "model.pkl"
META_PATH = "model_meta.json"
HISTORY_PATH = Path("prediction_log.csv")
PERFORMANCE_LOG_PATH = Path("performance_log.csv")

# ==========================================
# CONSTANTS — Feature list in exact training order
# ==========================================

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

# ==========================================
# Training distribution reference ranges
# ==========================================

FEATURE_RANGES = {
    "Store_TargetEnc": (5.0, 12.0),
    "CompetitionDistance": (0.0, 100_000.0),
    "CompetitionOpenMonths": (0.0, 600.0),
    "Promo2ActiveWeeks": (0.0, 260.0),
    "Week": (1, 53),
    "Year": (2013, 2030),
}