
# shap_interpretability.py

import streamlit as st
import shap
import matplotlib.pyplot as plt

try:
    from tpot_connector import latest_tpot_model, latest_X_train
except ImportError:
    latest_tpot_model = None
    latest_X_train = None


def run_shap_panel():
    st.subheader("🔍 SHAP Interpretability Panel")

    if latest_tpot_model is None or latest_X_train is None:
        st.warning("⚠️ No trained model found. Please run TPOT in the AutoML Launcher tab first.")
        return

    try:
        st.info("Generating SHAP summary plot for the latest TPOT model (sampled 100 rows)...")

        # Sample the training data for faster SHAP computation
        X_sample = latest_X_train.sample(n=min(100, len(latest_X_train)), random_state=42)

        # Use model directly — SHAP will auto-select best explainer
        explainer = shap.Explainer(latest_tpot_model, X_sample)
        shap_values = explainer(X_sample)

        st.markdown("### 📈 SHAP Summary Plot")
        fig, ax = plt.subplots()
        shap.plots.beeswarm(shap_values, max_display=10, show=False)
        st.pyplot(fig)

        st.markdown("### 💡 Top Feature Insights")
        top_features = shap_values.abs.mean(0).values
        sorted_indices = top_features.argsort()[::-1]
        feature_names = latest_X_train.columns[sorted_indices[:3]]

        st.markdown(f"**Top driver:** `{feature_names[0]}`")
        st.markdown(f"**Secondary factors:** `{feature_names[1]}`, `{feature_names[2]}`")

        st.success("✅ SHAP analysis completed successfully.")
    except Exception as e:
        st.error(f"❌ SHAP analysis failed. {type(e).__name__}: {e}")
