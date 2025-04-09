# synthetic_perturbation_tester.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from tpot_connector import _tpot_cache

def run_synthetic_perturbation_tester():
    st.title("🔬 Synthetic Perturbation Tester")

    model = _tpot_cache.get("latest_tpot_model")
    X_test = _tpot_cache.get("latest_X_test")
    y_test = _tpot_cache.get("latest_y_test")

    if model is None or X_test is None or y_test is None:
        st.warning("⚠️ Please run AutoML to generate models and test data.")
        return

    st.markdown("""
    This panel perturbs the test data to evaluate model stability under micro-changes.
    Select how much random noise to inject into numeric features.
    """)

    noise_pct = st.slider("💥 Percent Perturbation (e.g., 10 = ±10%)", 0, 50, 10)
    inject_missing = st.checkbox("🚫 Randomly Inject Missing Values", value=False)
    selected_features = st.multiselect("🎯 Features to Perturb", X_test.columns.tolist(), default=X_test.select_dtypes(include=np.number).columns.tolist())

    X_perturbed = X_test.copy()

    for col in selected_features:
        if pd.api.types.is_numeric_dtype(X_perturbed[col]):
            noise = X_perturbed[col] * (noise_pct / 100.0) * np.random.randn(len(X_perturbed))
            X_perturbed[col] += noise
        if inject_missing:
            mask = np.random.rand(len(X_perturbed)) < 0.05
            X_perturbed.loc[mask, col] = np.nan

    st.markdown("### 🔍 Preview of Perturbed Data")
    st.dataframe(X_perturbed.head())

    try:
        y_pred_original = model.predict(X_test)
        y_pred_perturbed = model.predict(X_perturbed)

        acc_original = accuracy_score(y_test, y_pred_original)
        acc_perturbed = accuracy_score(y_test, y_pred_perturbed)

        delta_accuracy = acc_perturbed - acc_original
        st.metric("📏 Accuracy Change", f"{delta_accuracy:.4f}", delta=f"{delta_accuracy:.4f}")

        if hasattr(model, "predict_proba"):
            y_proba_orig = model.predict_proba(X_test)
            y_proba_pert = model.predict_proba(X_perturbed)
            ll_orig = log_loss(y_test, y_proba_orig)
            ll_pert = log_loss(y_test, y_proba_pert)
            st.metric("📉 Log Loss Change", f"{ll_pert - ll_orig:.4f}")

    except Exception as e:
        st.error(f"❌ Perturbation evaluation failed: {type(e).__name__}: {e}")

    st.markdown("---")
    st.markdown("""
    ### 🧠 Interpretation
    - High sensitivity may indicate model brittleness.
    - Log loss shifts help expose confidence degradation.
    - Use this panel during validation and hardening phases.
    """)
