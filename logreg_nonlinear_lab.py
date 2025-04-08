# logreg_nonlinear_lab.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


def run_logreg_nonlinear_lab():
    st.header("🔀 LogReg Nonlinear Tricks")

    if "X" not in st.session_state or "y" not in st.session_state:
        st.warning("⚠️ Data not found in session. Please generate or load your data first.")
        return

    X = st.session_state.X.copy()
    y = st.session_state.y.copy()

    if not np.issubdtype(y.dtype, np.integer):
        st.info("ℹ️ Your target `y` does not appear to be binary or categorical. This module is best suited for classification.")

    st.markdown("""
    Apply nonlinear transformations and interaction terms to enhance your logistic regression models or help tree models discover nonlinearity.
    """)

    selected_cols = st.multiselect("📌 Select numeric columns to enhance", X.select_dtypes(include=np.number).columns.tolist())

    if not selected_cols:
        st.info("👈 Select at least one numeric column to proceed.")
        return

    new_X = X.copy()
    new_features = []

    for col in selected_cols:
        new_X[f"{col}_squared"] = X[col] ** 2
        new_X[f"{col}_log"] = np.log1p(X[col])
        new_X[f"{col}_inv"] = 1 / (X[col] + 1e-6)
        new_features.extend([f"{col}_squared", f"{col}_log", f"{col}_inv"])

    # Add interactions
    if len(selected_cols) >= 2:
        for i in range(len(selected_cols)):
            for j in range(i + 1, len(selected_cols)):
                c1, c2 = selected_cols[i], selected_cols[j]
                name = f"{c1}_x_{c2}"
                new_X[name] = X[c1] * X[c2]
                new_features.append(name)

    st.success(f"✅ Added {len(new_features)} new nonlinear & interaction features.")
    st.dataframe(new_X[new_features].head())

    # Option to update session state
    if st.checkbox("💾 Replace current X with this enhanced version"):
        st.session_state.X = new_X
        st.success("✅ Session state updated with enhanced features.")
