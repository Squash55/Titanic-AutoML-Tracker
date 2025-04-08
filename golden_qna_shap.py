# golden_qna_shap.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from tpot_connector import get_latest_model_and_data


def run_golden_qna_shap():
    st.header("🔮 Golden Q&A: SHAP-Powered Explanations")

    # ✅ Show model status if loaded from disk
    if "loaded_model" in st.session_state:
        st.info("✅ A previously saved model is currently active in memory.")

    # Load model and data
    model, X_train, y_train = get_latest_model_and_data()

    if model is None:
        st.warning("No model found. Please run AutoML first.")
        return

    # SHAP explainer setup
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_train)

    st.markdown("This module provides smart answers to golden questions using SHAP explanations.")

    row_idx = st.number_input("Select row for SHAP explanation:", min_value=0, max_value=len(X_train)-1, value=0)
    row_data = X_train.iloc[[row_idx]]
    shap_row = shap_values[row_idx]

    # Smart SHAP-based answer
    st.subheader("🤖 Smart SHAP-Based Answer")

    # Extract top contributing features
    shap_df = pd.DataFrame({
        'Feature': X_train.columns,
        'SHAP Value': shap_row.values,
        'Feature Value': row_data.values.flatten()
    })
    shap_df['Impact'] = shap_df['SHAP Value'].abs()
    top_features = shap_df.sort_values(by='Impact', ascending=False).head(3)

    # Answer template
    reasons = []
    for _, row in top_features.iterrows():
        direction = "increased" if row['SHAP Value'] > 0 else "decreased"
        reasons.append(f"'{row['Feature']}' = {row['Feature Value']} ({direction} prediction by {row['SHAP Value']:.2f})")

    st.markdown("This prediction was mainly influenced by: ")
    for r in reasons:
        st.markdown(f"- {r}")

    # Optional SHAP plot
    if st.checkbox("🔍 Show SHAP Waterfall Plot"):
        fig = shap.plots.waterfall(shap_values[row_idx], show=False)
        st.pyplot(fig)

    # Display data row for reference
    with st.expander("📄 Show Input Row Data"):
        st.dataframe(row_data.T, use_container_width=True)


# For standalone testing
if __name__ == '__main__':
    run_golden_qna_shap()
