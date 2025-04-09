# sensitivity_explorer.py

import streamlit as st
import pandas as pd
import numpy as np
from tpot_connector import _tpot_cache
def run_user_manual(deep=True, compact=True):
    
# full contents of your manual go here
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

    edge_case_mode = st.checkbox("🧪 Edge Case Mode", value=False)
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
                options.append("🚫 Missing")
            selected = st.selectbox(col, options)
            user_input[col] = None if selected == "🚫 Missing" else selected

    input_df = pd.DataFrame([user_input])
    st.markdown("### 🔍 Simulated Input")
    st.dataframe(input_df)

    # Save user input to session for PDF export
    for k, v in user_input.items():
        st.session_state[f"sens_input_{k}"] = v

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
    - Toggle **Edge Case Mode** to simulate real-world anomalies or stress tests.
    """)

    # ✅ Add session tracker for better navigation and PDF integration
    st.session_state["last_used_tab"] = "Sensitivity Explorer"
    st.session_state["include_sensitivity_pdf"] = st.sidebar.checkbox("🧪 Include Sensitivity Explorer Section", value=True)


# === Documentation Insertion for user_manual.py ===
def append_to_manual():
    st.markdown("### 📐 Sensitivity Explorer (What-if Panel)")
    st.markdown("""
    This panel allows you to simulate hypothetical scenarios by adjusting input feature values.

    - Use the **sliders and selectors** to create what-if inputs.
    - Toggle **Edge Case Mode** to test Min, Max, Random, or Missing values.
    - Get real-time predictions and probability scores based on your input.
    - Great for understanding how small changes affect outcomes.

    You can also include the current sensitivity configuration in the **PDF Report**.
    """)
