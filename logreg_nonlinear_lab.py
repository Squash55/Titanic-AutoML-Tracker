# logreg_nonlinear_lab.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_logreg_nonlinear_lab():
    st.header("🧠 LogReg Nonlinear Tricks")

    if "X" not in st.session_state:
        st.warning("❌ No dataset found. Please upload or generate data first.")
        return

    X = st.session_state.X.copy()
    st.subheader("🔧 Select Feature for Transformation")
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_columns:
        st.warning("⚠️ No numeric features available for transformation.")
        return

    selected_col = st.selectbox("Select numeric column", numeric_columns)
    transform = st.radio("Choose transformation type", ["Square (x²)", "Cube (x³)", "Log(x)", "1/x", "x * Other"], index=0)

    if transform == "Square (x²)":
        new_col = f"{selected_col}_squared"
        X[new_col] = X[selected_col] ** 2
        st.success(f"✅ Created: {new_col}")

    elif transform == "Cube (x³)":
        new_col = f"{selected_col}_cubed"
        X[new_col] = X[selected_col] ** 3
        st.success(f"✅ Created: {new_col}")

    elif transform == "Log(x)":
        new_col = f"log_{selected_col}"
        X[new_col] = np.log1p(X[selected_col])
        st.success(f"✅ Created: {new_col} (log1p)")

    elif transform == "1/x":
        new_col = f"inv_{selected_col}"
        X[new_col] = 1 / (X[selected_col].replace(0, np.nan))
        st.success(f"✅ Created: {new_col}")

    elif transform == "x * Other":
        other_col = st.selectbox("Multiply with", [col for col in numeric_columns if col != selected_col])
        new_col = f"{selected_col}_x_{other_col}"
        X[new_col] = X[selected_col] * X[other_col]
        st.success(f"✅ Created: {new_col}")

    st.session_state.X = X

    st.markdown("### 📊 Distribution of New Feature")
    fig, ax = plt.subplots()
    X[new_col].hist(ax=ax, bins=30)
    ax.set_title(f"Histogram of {new_col}")
    st.pyplot(fig)

    st.dataframe(X[[selected_col, new_col]].head())
