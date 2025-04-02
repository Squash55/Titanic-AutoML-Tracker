
import shap
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split

def run_shap_panel():
    st.header("🔍 SHAP + Interpretability Panel")
    st.subheader("📊 SHAP Summary Plot")

    df = pd.read_csv("sample_titanic_data.csv")
    df = df.dropna()
    X = pd.get_dummies(df.drop("Survived", axis=1))
    y = df["Survived"]

    model = xgb.XGBClassifier()
    model.fit(X, y)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.plots.bar(shap_values, show=False)
    st.pyplot()

    st.markdown("### 🧠 Smart Explanation")
    st.markdown("""
- **Sex**: The most powerful predictor. Females had much higher survival rates.
- **Fare**: Higher fare often meant better class and higher survival odds.
- **Pclass**: First-class passengers had better access to lifeboats and survived more.
""")
