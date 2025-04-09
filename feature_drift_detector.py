# feature_drift_detector.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from tpot_connector import _tpot_cache


def run_feature_drift_detector():
    st.title("📉 Feature Drift Detector")

    X_train = _tpot_cache.get("latest_X_train")
    X_new = st.session_state.get("X")  # Assume this is new uploaded/test data

    if X_train is None or X_new is None:
        st.warning("⚠️ Missing training or new dataset. Please run AutoML and upload or generate new data.")
        return

    shared_cols = [col for col in X_train.columns if col in X_new.columns]
    st.markdown(f"✅ Comparing {len(shared_cols)} shared features")

    drift_results = []

    for col in shared_cols:
        train_vals = X_train[col].dropna()
        new_vals = X_new[col].dropna()
        stat, p = ks_2samp(train_vals, new_vals)
        drift_results.append({
            "Feature": col,
            "KS Stat": round(stat, 3),
            "P-Value": round(p, 4),
            "Drift Flag": "⚠️ Likely Drift" if p < 0.05 else "✅ Stable"
        })

    result_df = pd.DataFrame(drift_results).sort_values("P-Value")
    st.dataframe(result_df, use_container_width=True)

    st.markdown("### 📊 Top Drifted Features")
    top_drift = result_df.head(10)
    for _, row in top_drift.iterrows():
        fig, ax = plt.subplots()
        sns.kdeplot(X_train[row["Feature"]], label="Train", fill=True, alpha=0.4)
        sns.kdeplot(X_new[row["Feature"]], label="New", fill=True, alpha=0.4)
        ax.set_title(f"{row['Feature']} (KS={row['KS Stat']}, p={row['P-Value']})")
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("""
    ### 🧠 Interpretation
    - Low p-values (< 0.05) indicate statistically significant distribution drift.
    - Use this panel to check whether your production inputs still resemble training data.
    - High drift = potential accuracy degradation or feature leakage.
    """)
