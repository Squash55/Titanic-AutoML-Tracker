import streamlit as st
import pandas as pd
import random
from tpot_connector import _tpot_cache

def generate_synthetic_scenarios(X, num_cases=5):
    scenarios = []
    for _ in range(num_cases):
        scenario = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                scenario[col] = round(random.uniform(X[col].min(), X[col].max()), 2)
            else:
                scenario[col] = random.choice(X[col].dropna().unique())
        scenarios.append(scenario)
    return pd.DataFrame(scenarios)

def run():
    st.title("🧪 AI-Generated Validation Scenarios")

    model = _tpot_cache.get("latest_tpot_model")
    X_train = _tpot_cache.get("latest_X_train")

    if model is None or X_train is None:
        st.warning("⚠️ TPOT model or training data not found. Please run AutoML first.")
        return

    st.markdown(""" 
    This module generates synthetic edge cases to test your model on uncommon or extreme scenarios.
    Useful for robustness checks, adversarial testing, and hypothesis generation.
    """)

    num_cases = st.slider("How many test cases to generate?", 3, 10, 5)
    X_synthetic = generate_synthetic_scenarios(X_train, num_cases=num_cases)

    st.markdown("### 🧪 Generated Scenarios")
    st.dataframe(X_synthetic)

    if st.button("⚙️ Predict on These Scenarios"):
        try:
            preds = model.predict(X_synthetic)
            X_synthetic["Prediction"] = preds
            st.success("✅ Model predictions computed.")
            st.dataframe(X_synthetic)
        except Exception as e:
            st.error(f"❌ Prediction failed: {type(e).__name__}: {e}")
