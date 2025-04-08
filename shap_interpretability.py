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

    # Use session-loaded model if available
    model = st.session_state.get("loaded_model", latest_tpot_model)

    if model is None or latest_X_train is None:
        st.warning("⚠️ No trained model or training data found. Please run AutoML or load a model.")
        return

    try:
        st.info("Generating SHAP summary plot (sampled 100 rows)...")

        X_sample = latest_X_train.sample(n=min(100, len(latest_X_train)), random_state=42)
        explainer = shap.Explainer(model, X_sample)
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

        if "loaded_model" in st.session_state:
            st.success("✅ SHAP analysis used a model loaded from disk.")
        else:
            st.success("✅ SHAP analysis used the latest TPOT model.")
    except Exception as e:
        st.error(f"❌ SHAP analysis failed. {type(e).__name__}: {e}")
