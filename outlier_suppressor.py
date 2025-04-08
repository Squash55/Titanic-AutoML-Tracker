# outlier_suppressor.py
import streamlit as st
import pandas as pd
import numpy as np


def detect_outliers_iqr(df):
    outlier_flags = pd.DataFrame(index=df.index)
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_flags[col] = ~df[col].between(lower, upper)
    return outlier_flags


def run_outlier_suppressor():
    st.title("🧹 Outlier Suppressor")

    if 'X' not in st.session_state:
        st.warning("⚠️ No data found. Please generate or upload data in the Synthetic Data Generator tab.")
        return

    df = st.session_state.X.copy()
    outliers = detect_outliers_iqr(df)
    total_outliers = outliers.sum().sum()
    st.info(f"Detected **{total_outliers}** total outliers across all numerical features.")

    method = st.radio("Choose Outlier Handling Method:", ["Cap Outliers", "Remove Rows", "Log Transform Affected Columns"])

    preview = df.copy()
    cols_to_process = outliers.columns[outliers.any()].tolist()

    if method == "Cap Outliers":
        for col in cols_to_process:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            preview[col] = df[col].clip(lower, upper)
    elif method == "Remove Rows":
        mask = ~outliers.any(axis=1)
        preview = df[mask]
    elif method == "Log Transform Affected Columns":
        for col in cols_to_process:
            preview[col] = np.log1p(df[col])

    st.subheader("📊 Preview Changes")
    st.dataframe(preview.head())

    if st.button("✅ Confirm and Save Cleaned Data"):
        st.session_state.X = preview
        st.success("Cleaned dataset has been stored in session_state.X!")
