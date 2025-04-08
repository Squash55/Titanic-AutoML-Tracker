# auto_fe_logreg_lab.py

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tpot_connector import _tpot_cache
from automl_launcher import run_automl_launcher

def run_logreg_interactions_explorer():
    st.title("🧪 LogReg + Interaction Explorer")

    X = st.session_state.get("X") or _tpot_cache.get("latest_X_train")
    y = st.session_state.get("y") or _tpot_cache.get("latest_y_train")

    if X is None or y is None:
        st.warning("⚠️ No dataset found. Please load data or run AutoML first.")
        if st.button("🚀 Launch AutoML Now"):
            run_automl_launcher()
        return

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
        y_pred = lr_model.predict(X_poly)

        st.success(f"✅ Logistic Regression AUC (with poly terms): **{auc:.3f}**")

        st.subheader("📈 Prediction Probability Histogram")
        fig, ax = plt.subplots()
        sns.histplot(y_proba, kde=True, bins=20, ax=ax, color="teal")
        ax.set_title("Predicted Probability Distribution (Train Set)")
        st.pyplot(fig)

        # Visualize confusion matrix
        cm = confusion_matrix(y, y_pred)
        st.subheader("🔍 Confusion Matrix")
        st.text(cm)

        # Save model as .pkl
        if st.button("💾 Export Model (.pkl)"):
            with open("logreg_poly_model.pkl", "wb") as f:
                pickle.dump(lr_model, f)
            st.success("✅ Model exported to logreg_poly_model.pkl")

    except Exception as e:
        st.error(f"❌ LogReg model failed: {type(e).__name__}: {e}")
