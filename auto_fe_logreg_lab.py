# auto_fe_logreg_lab.py

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def run_logreg_interactions_explorer():
    st.title("🧪 LogReg + Interaction Explorer")

    if "X" not in st.session_state or "y" not in st.session_state:
        st.warning("⚠️ No dataset found. Please load or generate synthetic data.")
        return

    X = st.session_state["X"]
    y = st.session_state["y"]

    st.markdown("This tool fits logistic regression models with interaction and polynomial terms, and displays model performance and p-values.")

    degree = st.slider("Polynomial Degree", 1, 3, 2)
    include_bias = st.checkbox("Include Bias Term", value=False)

    poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=include_bias)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)

    st.markdown(f"🧮 Total generated features: `{X_poly.shape[1]}`")

    try:
        logit_model = sm.Logit(y, sm.add_constant(X_poly)).fit(disp=0)
        summary = logit_model.summary2().tables[1]
        summary["Feature"] = ["const"] + list(feature_names)

        # Sort by p-value
        sorted_summary = summary.sort_values("P>|z|")
        top_n = st.slider("Top N terms to display", 5, 30, 10)

        st.subheader("📉 Top Terms by P-Value")
        st.dataframe(sorted_summary[["Feature", "Coef.", "P>|z|"]].head(top_n), use_container_width=True)

        overfit_flags = sorted_summary[sorted_summary["P>|z|"] > 0.05]
        if not overfit_flags.empty:
            st.warning(f"⚠️ {len(overfit_flags)} terms have p-values > 0.05 — may signal overfitting.")

        # Optional: Fit sklearn version to get AUC
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_poly, y)
        y_proba = lr_model.predict_proba(X_poly)[:, 1]
        auc = roc_auc_score(y, y_proba)
        st.success(f"✅ Logistic Regression AUC (with poly terms): **{auc:.3f}**")

        # Plot predicted probabilities
        st.subheader("📈 Prediction Distribution")
        fig, ax = plt.subplots()
        sns.histplot(y_proba, kde=True, bins=20, ax=ax, color="teal")
        ax.set_title("Predicted Probability Distribution (Train Set)")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ LogReg model failed: {type(e).__name__}: {e}")
