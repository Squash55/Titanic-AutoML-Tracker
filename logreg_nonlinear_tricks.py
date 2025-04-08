# logreg_nonlinear_tricks.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def run_logreg_nonlinear_tricks():
    st.title("📈 Logistic Regression + Nonlinear Feature Expansion")

    if "X" not in st.session_state or "y" not in st.session_state:
        st.warning("⚠️ No dataset found. Please generate or upload data in the Synthetic Data Generator tab first.")
        return

    X = st.session_state.X.copy()
    y = st.session_state.y.copy()

    if y.nunique() > 2:
        st.warning("🚫 Target must be binary for logistic regression.")
        return

    st.subheader("🔁 Polynomial Feature Expansion")
    degree = st.slider("Select polynomial degree", 1, 5, 2)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_expanded = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)

    st.write(f"Expanded features: {len(feature_names)}")

    X_train, X_test, y_train, y_test = train_test_split(X_expanded, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("🧪 Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("📊 Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    st.pyplot(fig)

    st.subheader("📌 Coefficient Summary")
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": model.coef_[0]})
    coef_df["Abs"] = np.abs(coef_df["Coefficient"])
    st.dataframe(coef_df.sort_values("Abs", ascending=False).drop(columns="Abs"))
