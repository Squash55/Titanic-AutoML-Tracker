# target_drift_diagnostic.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ks_2samp
from tpot_connector import _tpot_cache

def run_target_drift_diagnostic():
    st.title("🎯 Target Drift Diagnostic")

    y_train = _tpot_cache.get("latest_y_train")
    y_new = st.session_state.get("y")

    if y_train is None or y_new is None:
        st.warning("⚠️ Missing target variable for train or new data. Please run AutoML and ensure both are loaded.")
        return

    # Detect type: classification (discrete) vs regression (continuous)
    is_classification = y_train.nunique() <= 10

    st.markdown(f"Detected problem type: **{'Classification' if is_classification else 'Regression'}**")

    if is_classification:
        train_counts = y_train.value_counts().sort_index()
        new_counts = y_new.value_counts().reindex(train_counts.index, fill_value=0)

        df = pd.DataFrame({"Train": train_counts, "New": new_counts})
        chi2, p, _, _ = chi2_contingency(df.T)

        st.dataframe(df)
        st.metric("Chi-square P-Value", f"{p:.4f}")
        if p < 0.05:
            st.error("⚠️ Likely target drift detected!")
        else:
            st.success("✅ No significant target drift detected.")

        fig, ax = plt.subplots()
        df.plot.bar(ax=ax)
        ax.set_title("Target Class Distribution Comparison")
        st.pyplot(fig)

    else:
        stat, p = ks_2samp(y_train, y_new)

        st.metric("KS-Test P-Value", f"{p:.4f}")
        if p < 0.05:
            st.error("⚠️ Likely target drift detected!")
        else:
            st.success("✅ No significant target drift detected.")

        fig, ax = plt.subplots()
        sns.kdeplot(y_train, label="Train", fill=True, alpha=0.4)
        sns.kdeplot(y_new, label="New", fill=True, alpha=0.4)
        ax.set_title("Target Distribution (Regression)")
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("""
    ### 🧠 Interpretation
    - A significant change in the distribution of the target variable indicates that the **problem definition may be shifting**.
    - This can occur due to changes in business rules, label definitions, or external conditions.
    - Use this panel to decide whether to retrain or revalidate your model.
    """)
