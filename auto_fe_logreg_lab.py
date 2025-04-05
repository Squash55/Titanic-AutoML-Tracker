# auto_fe_logreg_lab.py
import streamlit as st
import pandas as pd
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from tpot_connector import _tpot_cache

def run_auto_fe_logreg_lab():
    st.title("🧪 Auto Feature Engineering + LogReg Lab")

    X = _tpot_cache.get("X_train")
    y = _tpot_cache.get("y_train")

    if X is None or y is None:
        st.warning("⚠️ Please run AutoML first to generate training data.")
        return

    max_degree = st.slider("Max Polynomial Degree", 1, 4, 2)
    include_interactions = st.checkbox("Include Interaction Terms", value=True)
    test_size = st.slider("Test Set Fraction", 0.1, 0.5, 0.2)

    st.markdown("---")
    st.markdown("### 🔍 Generated Feature Matrix")

    poly = PolynomialFeatures(degree=max_degree, interaction_only=not include_interactions, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names)
    st.dataframe(X_poly_df.head())

    st.markdown("---")
    st.markdown("### 🧠 Logistic Regression on Engineered Features")

    X_train, X_test, y_train, y_test = train_test_split(X_poly_df, y, test_size=test_size, random_state=42)

    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        st.success(f"✅ Accuracy on test set: {acc:.4f}")

        coef_df = pd.DataFrame({
            "Feature": X_poly_df.columns,
            "Coefficient": model.coef_[0]
        }).sort_values(by="Coefficient", key=abs, ascending=False)

        st.markdown("### 📊 Top Contributing Features")
        st.dataframe(coef_df.head(15))

    except Exception as e:
        st.error(f"❌ Logistic Regression failed: {type(e).__name__}: {e}")
