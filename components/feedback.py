import pandas as pd
import streamlit as st
from datetime import datetime

from config import FEEDBACK_LOG_PATH

# ==========================================
# Feedback Log
# ==========================================


def save_feedback(prediction: float, feedback: str) -> None:
    """
    Save a single feedback record to the feedback log CSV.

    Stores:
    - Timestamp (datetime.now())
    - The predicted sales value the feedback refers to
    - The feedback label ("Correct" or "Incorrect")

    Creates the feedback log file automatically if it does not exist,
    and appends new rows to it otherwise, without re-reading the
    existing file contents.
    """
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Predicted Sales (€)": round(prediction, 2),
        "Feedback": feedback,
    }

    new_record = pd.DataFrame([record])

    file_exists = FEEDBACK_LOG_PATH.exists()

    new_record.to_csv(
        FEEDBACK_LOG_PATH,
        mode="a",
        header=not file_exists,
        index=False,
    )


def load_feedback() -> pd.DataFrame:

    if FEEDBACK_LOG_PATH.exists():

        return pd.read_csv(FEEDBACK_LOG_PATH)

    return pd.DataFrame()


# ==========================================
# PREDICTION FEEDBACK DASHBOARD
# ==========================================

def _has_existing_feedback(prediction: float) -> bool:
    """
    Check feedback_log.csv directly for any existing feedback record
    (Correct or Incorrect) already tied to this prediction.

    This is CSV-level duplicate protection: it re-reads the log fresh
    each time it is called, independent of st.session_state, so it
    also catches the case where two clicks land before a rerun has
    had a chance to disable the buttons.
    """
    feedback = load_feedback()

    if feedback.empty:
        return False

    return bool((feedback["Predicted Sales (€)"] == round(prediction, 2)).any())


def render_feedback_dashboard(prediction: float) -> None:
    """
    Render the Prediction Feedback section for a given prediction.

    Displays two feedback buttons ("Correct" / "Incorrect") that each
    append one row to feedback_log.csv via save_feedback(), followed
    by a summary dashboard computed from the real feedback log.

    Duplicate submissions for the same prediction are prevented two ways:
    - st.session_state disables both buttons once feedback has been
      submitted in this session.
    - A fresh feedback_log.csv check runs before every save_feedback()
      call, so even a duplicate click that slips past the disabled
      button state is rejected instead of writing a second row.
    """

    st.divider()
    st.subheader("👍 Prediction Feedback")
    st.caption("Was this prediction helpful?")

    # --------------------------------------
    # Feedback Buttons
    # --------------------------------------

    feedback_key = f"feedback_submitted::{round(prediction, 2)}"
    already_submitted = st.session_state.get(feedback_key, False) or _has_existing_feedback(prediction)

    b1, b2 = st.columns(2)

    with b1:
        if st.button(
            "👍 Correct Prediction",
            use_container_width=True,
            disabled=already_submitted,
            key="feedback_btn_correct",
        ):
            if _has_existing_feedback(prediction):
                st.session_state[feedback_key] = True
                st.info("Feedback has already been recorded for this prediction.")
            else:
                save_feedback(prediction, "Correct")
                st.session_state[feedback_key] = True
                st.success("Thanks for your feedback!")

    with b2:
        if st.button(
            "👎 Incorrect Prediction",
            use_container_width=True,
            disabled=already_submitted,
            key="feedback_btn_incorrect",
        ):
            if _has_existing_feedback(prediction):
                st.session_state[feedback_key] = True
                st.info("Feedback has already been recorded for this prediction.")
            else:
                save_feedback(prediction, "Incorrect")
                st.session_state[feedback_key] = True
                st.warning("Thanks for your feedback — we'll use this to improve the model.")

    if already_submitted:
        st.caption("✅ Feedback already recorded for this prediction.")

    st.write("")

    # --------------------------------------
    # Feedback Summary
    # --------------------------------------

    st.markdown("### Feedback Summary")

    feedback = load_feedback()

    if feedback.empty:
        st.info("No feedback recorded yet. Be the first to rate a prediction!")
        return

    total_feedback = len(feedback)
    correct_feedback = (feedback["Feedback"] == "Correct").sum()
    incorrect_feedback = (feedback["Feedback"] == "Incorrect").sum()
    positive_rate = (correct_feedback / total_feedback) * 100

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Total Feedback", total_feedback)
    with m2:
        st.metric("Correct Feedback", int(correct_feedback))
    with m3:
        st.metric("Incorrect Feedback", int(incorrect_feedback))
    with m4:
        st.metric("Positive Feedback Rate", f"{positive_rate:.2f}%")