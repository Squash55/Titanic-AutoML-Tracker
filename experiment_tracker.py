import streamlit as st
import pandas as pd
from datetime import datetime
from tpot_connector import __dict__ as _tpot_cache

def run_experiment_tracker():
    st.subheader("📊 Experiment Tracker & CSV Export")

    if "experiment_log" not in _tpot_cache:
        _tpot_cache["experiment_log"] = []

    log = _tpot_cache["experiment_log"]

    if not log:
        st.info("📭 No experiments recorded yet. Run TPOT or RandomForest to begin tracking.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(log)
    df = df.sort_values(by="score", ascending=False)

    st.dataframe(df, use_container_width=True)

    # AI Insights Section
    st.markdown("### 🧠 AI Insights")
    if len(df) > 1:
        best_score = df.iloc[0]["score"]
        best_experiment = df.iloc[0]["experiment"]
        st.success(f"🎯 Best Experiment: **{best_experiment}** with a score of **{best_score:.4f}**")

        # Provide recommendations based on the results
        st.write("""
        **AI Recommendation**:
        - You might want to further fine-tune the hyperparameters of the top-performing experiments.
        - If the experiment with the highest score is based on a single model, try ensemble methods to improve stability and reduce overfitting.
        - Check the features used in the best models to ensure the data preprocessing step is optimal.
        """)

    # Allow the user to download the experiment log as a CSV file
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name="experiment_log.csv", mime="text/csv")
