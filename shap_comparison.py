# shap_comparison.py

import streamlit as st
import shap
import matplotlib.pyplot as plt

from tpot_connector import __dict__ as _tpot_cache


def run_shap_comparison():
    st.subheader("🧠 SHAP Comparison Across Models")

    tpot_model = _tpot_cache.get("latest_tpot_model")
    rf_model = _tpot_cache.get("latest_rf_model")
    X_train = _tpot_cache.get("latest_X_train")

    if X_train is None:
        st.warning("⚠️ Training data not found. Please run AutoML Comparison first.")
        return

    if not tpot_model and not rf_model:
        st.warning("⚠️ No models found. Train TPOT and/or RandomForest in the comparison panel.")
        return

    tab_option = st.radio("Select model to compare SHAP values:", ["TPOT", "RandomForest"], horizontal=True)

    if tab_option == "TPOT" and tpot_model:
        st.markdown("### 🔍 SHAP Summary for TPOT Model")
        try:
            explainer = shap.Explainer(tpot_model, X_train)
            shap_values = explainer(X_train)
            fig, ax = plt.subplots()
            shap.plots.beeswarm(shap_values, max_display=10, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ SHAP failed for TPOT: {type(e).__name__}: {e}")

    elif tab_option == "RandomForest" and rf_model:
        st.markdown("### 🌲 SHAP Summary for RandomForest")
        try:
            explainer = shap.Explainer(rf_model, X_train)
            shap_values = explainer(X_train)
            fig, ax = plt.subplots()
            shap.plots.beeswarm(shap_values, max_display=10, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ SHAP failed for RandomForest: {type(e).__name__}: {e}")
    else:
        st.warning("⚠️ Model not trained yet. Please run training first.")
