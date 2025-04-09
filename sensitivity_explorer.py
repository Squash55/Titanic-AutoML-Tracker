# sensitivity_explorer.py

import streamlit as st
import pandas as pd
import numpy as np
from tpot_connector import _tpot_cache


def run_sensitivity_explorer():
    st.title("📐 Sensitivity Explorer (What-if Panel)")

    model = _tpot_cache.get("latest_tpot_model")
    X_train = _tpot_cache.get("latest_X_train")

    if model is None or X_train is None:
        st.warning("⚠️ No model or training data found. Please run AutoML first.")
        return

    st.markdown("""
    Adjust each feature below to simulate hypothetical inputs.
    We'll show the model's prediction and probability (if available).
    """)

    user_input = {}
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            min_val = float(X_train[col].min())
            max_val = float(X_train[col].max())
            mean_val = float(X_train[col].mean())
            user_input[col] = st.slider(col, min_val, max_val, mean_val)
        else:
            options = list(X_train[col].dropna().unique())
            user_input[col] = st.selectbox(col, options)

    input_df = pd.DataFrame([user_input])
    st.markdown("### 🔍 Simulated Input")
    st.dataframe(input_df)

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🧠 Model Prediction: **{prediction}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            st.markdown("### 📈 Prediction Probabilities")
            proba_df = pd.DataFrame({"Class": model.classes_, "Probability": proba})
            st.dataframe(proba_df)
    except Exception as e:
        st.error(f"❌ Prediction failed: {type(e).__name__}: {e}")

    st.markdown("---")
    st.markdown("""
    ### 🧠 Interpretation
    - Use this panel to test edge cases and understand prediction drivers.
    - Try adjusting only one feature at a time to isolate sensitivity.
    - You can screenshot or log interesting scenarios for deeper analysis.
    """)
