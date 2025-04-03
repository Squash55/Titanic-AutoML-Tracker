
# shap_waterfall.py

import streamlit as st
import shap
import matplotlib.pyplot as plt

try:
    from tpot_connector import latest_tpot_model, latest_X_test
except ImportError:
    latest_tpot_model = None
    latest_X_test = None


def run_shap_waterfall():
    st.subheader("📉 SHAP Waterfall Plot (Individual Prediction)")

    if latest_tpot_model is None or latest_X_test is None:
        st.warning("⚠️ No trained model or test data found. Please run TPOT first.")
        return

    try:
        row_index = st.slider("Select Row Index", 0, len(latest_X_test) - 1, 0)
        row_data = latest_X_test.iloc[[row_index]]

        st.markdown("### 🧬 Selected Row Input")
        st.dataframe(row_data)

        explainer = shap.Explainer(latest_tpot_model, latest_X_test)
        shap_values = explainer(latest_X_test)

        st.markdown("### 🔎 SHAP Waterfall Plot")
        fig = shap.plots.waterfall(shap_values[row_index], show=False)
        st.pyplot(fig, clear_figure=True)

        st.success("✅ Waterfall plot generated for selected prediction.")
    except Exception as e:
        st.error(f"❌ SHAP waterfall plot failed. {type(e).__name__}: {e}")
