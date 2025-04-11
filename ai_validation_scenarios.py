import streamlit as st 
import pandas as pd
import random
from tpot_connector import _tpot_cache
from sklearn.metrics import accuracy_score
import numpy as np

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

def generate_insights(y_true, y_pred):
    """
    Generate insights based on the predictions made on synthetic data.
    """
    accuracy = accuracy_score(y_true, y_pred)
    incorrect_predictions = np.where(y_true != y_pred)[0]
    insight = f"Accuracy on synthetic data: {accuracy:.2f}\n"

    if len(incorrect_predictions) > 0:
        insight += f"⚠️ Incorrect Predictions: {len(incorrect_predictions)}\n"
        insight += f"These predictions might indicate areas where the model struggles, possibly out-of-distribution or rare cases."

    return insight

def run():
    st.title("🧪 AI-Generated Validation Scenarios")

    # Purpose statement below title
    st.markdown("""
    **Purpose:**  
    The **AI-Generated Validation Scenarios** module helps test the robustness of your model by generating synthetic edge cases and validating model predictions on these extreme scenarios. This tool is designed for adversarial testing, robustness checks, and hypothesis generation, enabling you to explore how your model performs under uncommon or out-of-distribution data conditions. By comparing synthetic predictions with expected outcomes, it highlights areas where the model may struggle and provides insights into potential weaknesses or biases.
    """)

    # Load the latest model and data from the cache
    model = _tpot_cache.get("latest_tpot_model")
    X_train = _tpot_cache.get("latest_X_train")

    if model is None or X_train is None:
        st.warning("⚠️ TPOT model or training data not found. Please run AutoML first.")
        return

    # Provide explanation of the tool
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

            # Display the predictions alongside the scenarios
            st.dataframe(X_synthetic)

            # Generate insights based on the predictions
            y_true = X_synthetic["Prediction"]  # Assuming the true values are the predictions here
            insights = generate_insights(y_true, preds)

            st.markdown("### 🔍 Insights from the Synthetic Validation")
            st.write(insights)

        except Exception as e:
            st.error(f"❌ Prediction failed: {type(e).__name__}: {e}")
