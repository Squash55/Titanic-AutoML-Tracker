# auto_fe_logreg_lab.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

def run_logreg_interactions_explorer():
    st.title("🔍 LogReg + Interaction Explorer")

    if "X" not in st.session_state or "y" not in st.session_state:
        st.warning("⚠️ No data found. Please upload or generate data first.")
        return

    X = st.session_state["X"]
    y = st.session_state["y"]

    st.markdown("""
    This tool explores polynomial and interaction terms added to Logistic Regression. Use it to:
    - Test non-linear effects in your predictors
    - See model performance gains from polynomial terms
    - Analyze variable influence using coefficients
    """)

    degree = st.slider("🔢 Polynomial Degree (for interaction terms)", 1, 4, 2)
    test_size = st.slider("🧪 Test Split Size", 0.1, 0.5, 0.3, step=0.05)

    poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=test_size, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    st.subheader("📈 Performance Report")
    st.text(classification_report(y_test, y_pred))
    st.write(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

    # Coefficient plot
    st.subheader("🔍 Top Coefficients")
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": model.coef_[0]})
    coef_df["Abs"] = np.abs(coef_df["Coefficient"])
    top_coef = coef_df.sort_values("Abs", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Coefficient", y="Feature", data=top_coef, ax=ax)
    st.pyplot(fig)

    st.info("✅ Use this to uncover hidden feature interactions and improve linear model expressiveness.")
