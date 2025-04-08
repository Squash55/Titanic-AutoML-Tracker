# logreg_nonlinear_lab.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_logreg_nonlinear_lab():
    st.header("🔁 LogReg Nonlinear Tricks Lab")

    if "X" not in st.session_state or st.session_state.X is None:
        st.warning("❌ No data found in session. Please upload or generate synthetic data first.")
        return

    df = st.session_state.X.copy()
    st.markdown("Select numeric columns to engineer non-linear features:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.warning("⚠️ No numeric columns available for transformation.")
        return

    selected_cols = st.multiselect("Choose feature(s)", numeric_cols, default=numeric_cols[:1])
    transforms = st.multiselect("Choose transformations", ["x^2", "x^3", "log(x)", "1/x"], default=["x^2"])

    if selected_cols and transforms:
        for col in selected_cols:
            x = df[col].replace(0, 1e-6)  # avoid log(0) or div by zero
            if "x^2" in transforms:
                df[f"{col}^2"] = x ** 2
            if "x^3" in transforms:
                df[f"{col}^3"] = x ** 3
            if "log(x)" in transforms:
                df[f"log_{col}"] = np.log(np.abs(x))
            if "1/x" in transforms:
                df[f"1_over_{col}"] = 1.0 / x

    # Interactions
    if len(selected_cols) > 1:
        st.markdown("#### ➕ Add interaction terms?")
        add_interactions = st.checkbox("Create pairwise interactions")
        if add_interactions:
            for i in range(len(selected_cols)):
                for j in range(i + 1, len(selected_cols)):
                    new_col = f"{selected_cols[i]}_x_{selected_cols[j]}"
                    df[new_col] = df[selected_cols[i]] * df[selected_cols[j]]

    st.success("✅ Nonlinear transformations applied.")
    st.session_state.X = df  # update the main session X

    st.subheader("📊 Preview of Engineered Features")
    preview_cols = [c for c in df.columns if any(s in c for s in ["^2", "^3", "log_", "1_over", "_x_"])]
    st.dataframe(df[preview_cols].head())

    st.subheader("📈 Histogram of One Transformed Feature")
    if preview_cols:
        selected_hist = st.selectbox("Select column for histogram", preview_cols)
        fig, ax = plt.subplots()
        df[selected_hist].hist(ax=ax, bins=30, color="skyblue")
        ax.set_title(f"Distribution of {selected_hist}")
        st.pyplot(fig)
