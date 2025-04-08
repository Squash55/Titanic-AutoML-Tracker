# catreg_switcher.py
import streamlit as st
import pandas as pd
import numpy as np

def run_catreg_switcher():
    st.title("🔀 Categorical ↔ Regression Switcher")

    if "X" not in st.session_state or "y" not in st.session_state:
        st.warning("⚠️ No dataset found. Please load or generate data first.")
        return

    df = st.session_state.X.copy()
    df["Target"] = st.session_state.y
    selected_column = st.selectbox("Select a column to transform", df.columns)

    conversion_type = st.radio("Choose conversion direction:",
                                ["Categorical → Numeric (Encoding)", "Numeric → Categorical (Binning)"])

    if conversion_type == "Categorical → Numeric (Encoding)":
        method = st.selectbox("Encoding method", ["Ordinal", "One-Hot"])

        if df[selected_column].dtype == object or df[selected_column].dtype.name == "category":
            if method == "Ordinal":
                mapping = {k: v for v, k in enumerate(sorted(df[selected_column].unique()))}
                st.write("Encoding Map:", mapping)
                df[selected_column] = df[selected_column].map(mapping)
            else:
                df = pd.get_dummies(df, columns=[selected_column], drop_first=True)

            st.success("✅ Column encoded. Preview below:")
            st.dataframe(df.head())

        else:
            st.warning("⚠️ Column must be categorical.")

    else:  # Numeric → Categorical
        bin_method = st.selectbox("Binning method", ["Quantile", "Equal-width"])
        num_bins = st.slider("Number of bins", 2, 10, 4)

        if np.issubdtype(df[selected_column].dtype, np.number):
            if bin_method == "Quantile":
                df[selected_column + "_binned"] = pd.qcut(df[selected_column], q=num_bins, labels=False)
            else:
                df[selected_column + "_binned"] = pd.cut(df[selected_column], bins=num_bins, labels=False)

            st.success("✅ Column binned. Preview below:")
            st.dataframe(df[[selected_column, selected_column + "_binned"]].head())

        else:
            st.warning("⚠️ Column must be numeric to bin it.")

    # Offer overwrite session option
    if st.button("🔁 Update session_state.X with transformed data"):
        df.drop(columns=["Target"], errors="ignore", inplace=True)
        st.session_state.X = df
        st.success("✅ session_state.X updated with transformations!")
