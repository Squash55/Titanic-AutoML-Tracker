# catreg_switcher.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_catreg_switcher():
    st.header("🔁 Cat ↔ Reg Switcher")

    st.markdown("""
    ### 🧠 What This Tool Does
    This switcher helps you **convert a column between categorical and regression formats**, depending on the modeling needs.

    - **Categorical ➜ Regression**: Useful for algorithms that only accept numerical inputs.
    - **Regression ➜ Categorical**: Handy when you want to group continuous values into bins (e.g., for classification tasks or segment-based analysis).

    The tool supports **manual binning** and **quantile-based binning**, plus lets you **restore the original column** anytime.
    """)

    if "X" not in st.session_state or "y" not in st.session_state:
        st.warning("❌ No data found in session. Please upload or generate synthetic data first.")
        return

    df = st.session_state.X.copy()
    df["Target"] = st.session_state.y.copy()

    column_to_convert = st.selectbox("🔽 Select Column to Transform", df.columns, index=len(df.columns) - 1)
    direction = st.radio("➡️ Select Conversion Direction", ["Categorical ➜ Regression", "Regression ➜ Categorical"])

    backup_key = f"backup_{column_to_convert}"
    if backup_key not in st.session_state:
        st.session_state[backup_key] = df[column_to_convert].copy()

    if direction == "Categorical ➜ Regression":
        st.subheader("🎯 Convert Categorical to Numeric Regression")

        if df[column_to_convert].dtype.name == "category" or df[column_to_convert].dtype == object:
            numeric_map = {k: i for i, k in enumerate(sorted(df[column_to_convert].unique()))}
            df[column_to_convert] = df[column_to_convert].map(numeric_map)
            st.success("✅ Converted categorical values to regression-friendly format.")
            st.code(numeric_map, language="python")
        else:
            st.info("This column is already numeric. No conversion needed.")

    elif direction == "Regression ➜ Categorical":
        st.subheader("🧮 Convert Numeric Column to Categorical")
        bin_mode = st.radio("Choose Binning Method", ["Manual Binning", "Quantile Binning"], index=1)

        if bin_mode == "Manual Binning":
            bins = st.slider("Number of Categories (Bins)", 2, 10, 3)
            binned = pd.cut(df[column_to_convert], bins=bins, labels=[f"Class_{i+1}" for i in range(bins)])
        else:
            bins = st.slider("Number of Quantiles", 2, 10, 4)
            binned = pd.qcut(df[column_to_convert], q=bins, labels=[f"Q{i+1}" for i in range(bins)], duplicates='drop')

        df[column_to_convert] = binned
        st.success(f"✅ Converted to categorical using {bins} bins.")

        st.dataframe(pd.DataFrame({"Original": st.session_state[backup_key], "Transformed": df[column_to_convert]}).head())

        # Histogram preview
        fig, ax = plt.subplots(figsize=(6, 3))
        df[column_to_convert].value_counts().sort_index().plot(kind="bar", ax=ax)
        ax.set_title("Distribution of Binned Classes")
        st.pyplot(fig)

    # Save updated column to session
    st.session_state.X[column_to_convert] = df[column_to_convert]
    if column_to_convert == "Target":
        st.session_state.y = df[column_to_convert]

    # Restore option
    if st.button(f"♻️ Restore Original '{column_to_convert}'"):
        st.session_state.X[column_to_convert] = st.session_state[backup_key]
        if column_to_convert == "Target":
            st.session_state.y = st.session_state[backup_key]
        st.success(f"✅ Restored original version of '{column_to_convert}'.")
