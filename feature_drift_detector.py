# feature_drift_detector.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency
from tpot_connector import _tpot_cache


def run_feature_drift_detector():
    st.title("🔍 Feature Drift Detector")

    X_train = _tpot_cache.get("latest_X_train")
    X_new = st.session_state.get("X")

    if X_train is None or X_new is None:
        st.warning("⚠️ Training or new input data not found. Please run AutoML and load new data.")
        return

    common_cols = [col for col in X_train.columns if col in X_new.columns]
    st.markdown(f"Found **{len(common_cols)}** overlapping columns between train and new data.")

    drift_results = []

    for col in common_cols:
        try:
            train_vals = X_train[col].dropna()
            new_vals = X_new[col].dropna()

            if pd.api.types.is_numeric_dtype(train_vals):
                stat, p = ks_2samp(train_vals, new_vals)
                test_name = "KS"
            else:
                train_counts = train_vals.value_counts()
                new_counts = new_vals.value_counts()
                categories = sorted(set(train_counts.index) | set(new_counts.index))
                train_freq = [train_counts.get(c, 0) for c in categories]
                new_freq = [new_counts.get(c, 0) for c in categories]
                stat, p = chi2_contingency([train_freq, new_freq])[:2]
                test_name = "Chi²"

            drift_results.append({
                "Feature": col,
                "Test": test_name,
                "P-Value": round(p, 4),
                "Drift": "⚠️ Yes" if p < 0.05 else "✅ No"
            })

        except Exception as e:
            drift_results.append({
                "Feature": col,
                "Test": "Error",
                "P-Value": "N/A",
                "Drift": f"❌ {type(e).__name__}"
            })

    df_drift = pd.DataFrame(drift_results)
    st.markdown("### 📋 Drift Summary")
    st.dataframe(df_drift, use_container_width=True)

    # Download CSV
    csv = df_drift.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Drift Report", data=csv, file_name="feature_drift_report.csv", mime="text/csv")

    # Visualize flagged drift
    st.markdown("### 📊 Selected Drifted Feature Distribution")
    drifted_cols = df_drift[df_drift["Drift"] == "⚠️ Yes"]["Feature"].tolist()

    if drifted_cols:
        col_choice = st.selectbox("Select a drifted feature to visualize", drifted_cols)
        if pd.api.types.is_numeric_dtype(X_train[col_choice]):
            fig, ax = plt.subplots()
            sns.kdeplot(X_train[col_choice], label="Train", fill=True, alpha=0.3)
            sns.kdeplot(X_new[col_choice], label="New", fill=True, alpha=0.3)
            ax.set_title(f"Distribution of {col_choice}")
            ax.legend()
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots()
            train_counts = X_train[col_choice].value_counts()
            new_counts = X_new[col_choice].value_counts()
            df_bar = pd.DataFrame({"Train": train_counts, "New": new_counts}).fillna(0)
            df_bar.plot.bar(ax=ax)
            ax.set_title(f"Category Distribution: {col_choice}")
            st.pyplot(fig)

    st.markdown("---")
    st.markdown("""
    ### 🧠 Interpretation
    - Drift may indicate that **your model's assumptions no longer hold**.
    - For severe drift, consider retraining or monitoring predictions on these features.
    - Use drifted features to trigger alerts or run fairness audits.
    """)
