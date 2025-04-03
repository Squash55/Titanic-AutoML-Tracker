# experiment_tracker.py

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

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name="experiment_log.csv", mime="text/csv")