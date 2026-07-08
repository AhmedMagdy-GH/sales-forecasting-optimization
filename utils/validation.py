

from config import FEATURE_RANGES

# ==========================================
# INPUT VALIDATION
# ==========================================

def validate_inputs(
    store_target_enc: float,
    competition_distance: float,
    competition_distance_missing: int,
    competition_open_months: float,
    competition_open_missing: int,
    promo2_active_weeks: float,
    week: int,
    year: int,
) -> list[str]:
    """
    Return a list of human-readable warning strings for inputs that fall
    outside the ranges seen during training.  An empty list means all
    inputs are within expected bounds.
    """
    warnings_out = []

    lo, hi = FEATURE_RANGES["Store_TargetEnc"]
    if not (lo <= store_target_enc <= hi):
        warnings_out.append(
            f"Store Target Encoding ({store_target_enc:.4f}) is outside the "
            f"training range ({lo}–{hi}).  Predictions may be unreliable."
        )

    lo, hi = FEATURE_RANGES["CompetitionDistance"]
    if not (lo <= competition_distance <= hi):
        warnings_out.append(
            f"Competition Distance ({competition_distance:,.0f} m) is outside "
            f"the expected range ({lo:,.0f}–{hi:,.0f} m)."
        )

    if competition_distance < 0:
        warnings_out.append("Competition Distance cannot be negative.")

    lo, hi = FEATURE_RANGES["CompetitionOpenMonths"]
    if not (lo <= competition_open_months <= hi):
        warnings_out.append(
            f"Competition Open Months ({competition_open_months}) is unusually "
            f"large (expected ≤ {hi}).  Please verify the value."
        )

    lo, hi = FEATURE_RANGES["Promo2ActiveWeeks"]
    if not (lo <= promo2_active_weeks <= hi):
        warnings_out.append(
            f"Promo2 Active Weeks ({promo2_active_weeks}) is outside the "
            f"expected range ({lo}–{hi})."
        )

    lo, hi = FEATURE_RANGES["Week"]
    if not (lo <= week <= hi):
        warnings_out.append(
            f"Week ({week}) must be between {lo} and {hi}."
        )

    lo, hi = FEATURE_RANGES["Year"]
    if not (lo <= year <= hi):
        warnings_out.append(
            f"Year ({year}) must be between {lo} and {hi}."
        )
    # Check for inconsistent competition distance inputs
    if competition_distance_missing == 1 and competition_distance > 0:
        warnings_out.append(
            "Competition Distance is provided, so Competition Distance Missing should be 0."
        )

    # Check for inconsistent competition open inputs
    if competition_open_missing == 1 and competition_open_months > 0:
        warnings_out.append(
            "Competition Open Months is provided, so Competition Open Missing should be 0."
        )

    return warnings_out
