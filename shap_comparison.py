
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
        st.warning("⚠️ Training data not found. Please run a model from AutoML Comparison or Launcher first.")
        return

    if not tpot_model and not rf_model:
        st.warning("⚠️ No models found. Please train TPOT and/or RandomForest.")
        return

    if tpot_model:
        st.markdown("### 🔍 SHAP Summary for TPOT Model")
        try:
            explainer = shap.Explainer(tpot_model, X_train)
            shap_values = explainer(X_train)
            fig, ax = plt.subplots()
            shap.plots.beeswarm(shap_values, max_display=10, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ TPOT SHAP failed: {type(e).__name__}: {e}")

    if rf_model:
        st.markdown("### 🌲 SHAP Summary for RandomForest")
        try:
            explainer = shap.Explainer(rf_model, X_train)
            shap_values = explainer(X_train)
            fig, ax = plt.subplots()
            shap.plots.beeswarm(shap_values, max_display=10, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ RandomForest SHAP failed: {type(e).__name__}: {e}")

    if tpot_model and rf_model:
        st.success("✅ SHAP comparisons for both models complete.")
