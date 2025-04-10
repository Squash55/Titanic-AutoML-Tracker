# user_manual.py

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
from tpot_connector import _tpot_cache

def run_sensitivity_explorer():
    st.title("\U0001F4D0 Sensitivity Explorer (What-if Panel)")

    model = _tpot_cache.get("latest_tpot_model")
    X_train = _tpot_cache.get("latest_X_train")

    if model is None or X_train is None:
        st.warning("\u26a0\ufe0f No model or training data found. Please run AutoML first.")
        return

    st.markdown("""
    Adjust each feature below to simulate hypothetical inputs.
    We'll show the model's prediction and probability (if available).
    """)

    edge_case_mode = st.checkbox("\U0001F9EA Edge Case Mode", value=False)
    user_input = {}

    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            min_val = float(X_train[col].min())
            max_val = float(X_train[col].max())
            mean_val = float(X_train[col].mean())

            if edge_case_mode:
                extreme_option = st.radio(
                    f"{col} (Extreme Mode)",
                    ["Normal", "Min", "Max", "Random"],
                    horizontal=True
                )
                if extreme_option == "Min":
                    val = min_val
                elif extreme_option == "Max":
                    val = max_val
                elif extreme_option == "Random":
                    val = float(np.random.uniform(min_val, max_val))
                else:
                    val = mean_val
                user_input[col] = val
            else:
                user_input[col] = st.slider(col, min_val, max_val, mean_val)
        else:
            options = list(X_train[col].dropna().unique())
            if edge_case_mode:
                options.append("\ud83d\udeab Missing")
            selected = st.selectbox(col, options)
            user_input[col] = None if selected == "\ud83d\udeab Missing" else selected

    input_df = pd.DataFrame([user_input])
    st.markdown("### \U0001F50D Simulated Input")
    st.dataframe(input_df)

    for k, v in user_input.items():
        st.session_state[f"sens_input_{k}"] = v

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"\U0001F9E0 Model Prediction: **{prediction}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            st.markdown("### \U0001F4C8 Prediction Probabilities")
            proba_df = pd.DataFrame({"Class": model.classes_, "Probability": proba})
            st.dataframe(proba_df)
    except Exception as e:
        st.error(f"\u274c Prediction failed: {type(e).__name__}: {e}")

    st.markdown("---")
    st.markdown("""
    ### \U0001F9E0 Interpretation
    - Use this panel to test edge cases and understand prediction drivers.
    - Try adjusting only one feature at a time to isolate sensitivity.
    - Toggle **Edge Case Mode** to simulate real-world anomalies or stress tests.
    
    **Pro Tips:**
    - Use this as a sandbox before real deployment — edge inputs often expose model fragility.
    - Combine with SHAP force plots to confirm feature influence changes match expectations.
    """)

    st.session_state["last_used_tab"] = "Sensitivity Explorer"
    st.session_state["include_sensitivity_pdf"] = st.sidebar.checkbox("\U0001F9EA Include Sensitivity Explorer Section", value=True)

def run_user_manual():
    st.title("\U0001F4D8 DAIVID Analytics User Manual")

    with st.expander("\U0001F680 How to Use This App", expanded=True):
        st.markdown("""
        Welcome to **DAIVID** – your AI co-pilot for predictive modeling!

        1. Start with **Auto EDA** to explore your dataset
        2. Launch **AutoML** to automatically build a model
        3. Use **SHAP**, **DOE**, and **Residual Plots** to interpret it
        4. Export everything to a **PDF Report**
        """)

    st.markdown("---")
    st.header("\U0001F4D6 Module Reference Guide")

    with st.expander("\U0001F4D0 Sensitivity Explorer (What-if Panel)"):
        st.markdown("""
        Simulate hypothetical scenarios by adjusting feature inputs.

        - Sliders and selectors for each feature
        - Edge Case Mode for extreme values or missingness
        - Get real-time predictions and probabilities

        **Pro Tips:**
        - Run multiple simulations and compare them visually using SHAP.
        - Use this to build adversarial test cases.
        """)
        if st.session_state.get("manual_image_mode"):
            st.image("screenshots/sensitivity_explorer_demo.png", caption="Sensitivity Explorer Screenshot", use_column_width=True)

    # Additional modules will be inserted below in sequence
