# shap_permutation_delta.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from tpot_connector import _tpot_cache


def run_shap_permutation_delta():
    st.header("📊 SHAP vs Permutation Importance Delta Viewer")

    model = _tpot_cache.get("model")
    X_test = _tpot_cache.get("X_test")
    y_test = _tpot_cache.get("y_test")

    if model is None or X_test is None or y_test is None:
        st.warning("❌ TPOT model or test data not found.")
        return

    with st.expander("ℹ️ Why this matters", expanded=True):
        st.markdown("""
        Comparing **SHAP values** and **permutation importance** helps identify when features
        are interacting, redundant, or possibly leaking information.

        - **SHAP**: Local explanations, model-specific
        - **Permutation**: Global, model-agnostic

        Large differences between the two may indicate **unstable feature behavior**.
        """)

    st.markdown("### 🔍 Step 1: Compute SHAP and Permutation Importances")

    explainer = shap.Explainer(model.predict, X_test)
    shap_values = explainer(X_test)
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame({"Feature": X_test.columns, "SHAP Importance": shap_importance})

    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({"Feature": X_test.columns, "Permutation Importance": result.importances_mean})

    merged = pd.merge(shap_df, perm_df, on="Feature")
    merged["Delta"] = np.abs(merged["SHAP Importance"] - merged["Permutation Importance"])
    merged = merged.sort_values("SHAP Importance", ascending=False)

    st.markdown("### 📈 Step 2: Compare and Visualize")
    fig, ax = plt.subplots(figsize=(10, 6))
    merged.set_index("Feature")[
        ["SHAP Importance", "Permutation Importance"]
    ].plot(kind="bar", ax=ax)
    plt.title("SHAP vs Permutation Importances")
    plt.ylabel("Importance")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### 🚨 Step 3: Flag Large Discrepancies")
    high_delta = merged[merged["Delta"] > 0.05]
    if not high_delta.empty:
        st.warning("⚠️ Features with large SHAP vs Permutation gaps:")
        st.dataframe(high_delta[["Feature", "SHAP Importance", "Permutation Importance", "Delta"]])
    else:
        st.success("✅ No major discrepancies found.")

    st.markdown("### 📎 Full Importance Table")
    st.dataframe(merged.reset_index(drop=True))

    # Store for PDF
    fig_path = "shap_perm_delta_plot.png"
    fig.savefig(fig_path)
    st.session_state["include_shap_perm_delta_pdf"] = st.checkbox("📥 Include in PDF Report")
    st.session_state["shap_perm_delta_plot_path"] = fig_path
