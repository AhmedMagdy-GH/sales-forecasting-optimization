import pandas as pd
import streamlit as st

# ==========================================
# BATCH PREDICTION — Upload Interface
# ==========================================

def render_batch_prediction() -> None:
    """Render the Batch Prediction upload interface (upload + preview only)."""

    st.divider()
    st.subheader("📂 Batch Prediction")
    st.caption("Upload a CSV file for batch prediction.")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:
            batch_df = pd.read_csv(uploaded_file)

            st.success(
                f"✅ Rows: {batch_df.shape[0]} | Columns: {batch_df.shape[1]}"
            )
            st.caption(
                f"{batch_df.shape[1]} columns detected."
            )
            st.write(
                f"**Uploaded File:** {uploaded_file.name}"
            )
            st.markdown("### Preview")
            st.dataframe(
                batch_df.head(10),
                use_container_width=True,
                hide_index=True,
            )

        except Exception as exc:
            st.error(f"❌ Failed to read CSV: {exc}")

    st.button(
        "🚀 Run Batch Prediction",
        use_container_width=True,
        disabled=True,
    )