import streamlit as st

# ==========================================
# INPUT FORM — Store Information
# ==========================================

def store_form() -> dict:
    """Render the Store Information section and return the collected values."""
    st.subheader("📊 Store Information")

    c1, c2 = st.columns(2)

    with c1:

        Store_TargetEnc = st.number_input(
            "Store Target Encoding",
            value=8.50,
            format="%.4f",
            help="Target-encoded store identifier derived from historical mean sales."
        )

        CompetitionDistance = st.number_input(
            "Competition Distance (m)",
            value=500.0,
            min_value=0.0,
            help="Distance in metres to the nearest competitor store."
        )

        CompetitionDistanceMissing = st.selectbox(
            "Competition Distance Missing",
            [0, 1],
            help="1 if CompetitionDistance was originally missing in the raw data."
        )

        CompetitionOpenMissing = st.selectbox(
            "Competition Open Missing",
            [0, 1],
            help="1 if CompetitionOpenSince fields were originally missing."
        )

    with c2:

        DayOfWeek = st.selectbox(
            "Day Of Week",
            [1, 2, 3, 4, 5, 6, 7],
            index=4,
            help="1 = Monday … 7 = Sunday."
        )
        # Automatically derive IsWeekend from the selected DayOfWeek.
        IsWeekend = 1 if DayOfWeek in [6, 7] else 0

        StoreType = st.selectbox(
            "Store Type",
            [0, 1, 2, 3],
            help="Label-encoded store category (a=0, b=1, c=2, d=3)."
        )

        Assortment = st.selectbox(
            "Assortment",
            [0, 1, 2],
            help="Label-encoded assortment level (a=0, b=1, c=2)."
        )

    return {
        "Store_TargetEnc": Store_TargetEnc,
        "CompetitionDistance": CompetitionDistance,
        "CompetitionDistanceMissing": CompetitionDistanceMissing,
        "CompetitionOpenMissing": CompetitionOpenMissing,
        "DayOfWeek": DayOfWeek,
        "IsWeekend": IsWeekend,
        "StoreType": StoreType,
        "Assortment": Assortment,
    }


# ==========================================
# INPUT FORM — Promotion Information
# ==========================================

def promotion_form() -> dict:
    """Render the Promotion Information section and return the collected values."""
    st.divider()
    st.subheader("🎁 Promotion Information")

    c1, c2 = st.columns(2)

    with c1:

        Promo = st.selectbox(
            "Promo",
            [0, 1],
            help="1 if the store is running a promotion on this day."
        )

        Promo2 = st.selectbox(
            "Promo2",
            [0, 1],
            help="1 if the store participates in the continuous Promo2 promotion."
        )

        Promo2ActiveWeeks = st.number_input(
            "Promo2 Active Weeks",
            value=0.0,
            min_value=0.0,
            help="Number of weeks since Promo2 started for this store."
        )

    with c2:

        SchoolHoliday = st.selectbox(
            "School Holiday",
            [0, 1],
            help="1 if the date falls within a school holiday period."
        )

        StateHoliday = st.selectbox(
            "State Holiday",
            [0, 1],
            help="1 if the date is a public / state holiday."
        )

        IsPromo2Active = st.selectbox(
            "Promo2 Active",
            [0, 1],
            help="1 if Promo2 is currently active for this store on this date."
        )

    return {
        "Promo": Promo,
        "Promo2": Promo2,
        "Promo2ActiveWeeks": Promo2ActiveWeeks,
        "SchoolHoliday": SchoolHoliday,
        "StateHoliday": StateHoliday,
        "IsPromo2Active": IsPromo2Active,
    }


# ==========================================
# INPUT FORM — Calendar Information
# ==========================================

def calendar_form() -> dict:
    """Render the Calendar Information section and return the collected values."""
    st.divider()
    st.subheader("📅 Calendar Information")

    c1, c2 = st.columns(2)

    with c1:

        Year = st.number_input(
            "Year",
            min_value=2013,
            max_value=2030,
            value=2015,
            help="Calendar year of the forecast date."
        )

        Week = st.number_input(
            "Week",
            min_value=1,
            max_value=53,
            value=30,
            help="ISO calendar week number (1–53)."
        )

    with c2:

        IsMonthStart = st.selectbox(
            "Month Start",
            [0, 1],
            help="1 if this is the first day of the month."
        )

        IsMonthEnd = st.selectbox(
            "Month End",
            [0, 1],
            help="1 if this is the last day of the month."
        )

    return {
        "Year": Year,
        "Week": Week,
        "IsMonthStart": IsMonthStart,
        "IsMonthEnd": IsMonthEnd,
    }


# ==========================================
# INPUT FORM — Competition Information
# ==========================================

def competition_form() -> dict:
    """Render the Competition Information section and return the collected values."""
    st.divider()
    st.subheader("🏢 Competition Information")

    c1, c2 = st.columns(2)

    with c1:

        CompetitionOpenMonths = st.number_input(
            "Competition Open Months",
            value=12.0,
            min_value=0.0,
            help="Number of months the nearest competitor has been open."
        )

    with c2:

        st.info(
            "Enter the competition information before generating the prediction."
        )

    return {
        "CompetitionOpenMonths": CompetitionOpenMonths,
    }