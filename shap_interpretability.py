import streamlit as st
import shap
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

def run_shap_panel():
    st.title("🔍 SHAP + Interpretability Panel")
    st.subheader("📊 SHAP Summary Plot")

    # Load sample Titanic dataset (or adapt to your actual dataset)
    df = pd.read_csv("sample_titanic_data.csv")

    # Simple preprocessing
    df = df.dropna()
    df = pd.get_dummies(df, columns=["Sex"], drop_first=False)
    features = ["Sex_male", "Sex_female", "Fare", "Age", "Pclass", "PassengerId"]
    X = df[features]
    y = df["Survived"]

    # Train a simple model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)

    # Compute SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Generate SHAP summary plot using future-proof method
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig)

    # Optional: smart explanation
    st.subheader("🧠 Smart Explanation")
    st.markdown("**Sex**: The most powerful predictor. Females had much higher survival rates.")
    st.markdown("**Fare**: Higher fare often meant better class and higher survival odds.")
    st.markdown("**Pclass**: First-class passengers had better access to lifeboats and survived more.")
