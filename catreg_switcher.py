# catreg_switcher.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder


def run_catreg_switcher():
    st.title("🔁 CatReg Switcher")
    st.markdown("Switch between regression and classification targets seamlessly.")

    if "y" not in st.session_state:
        st.warning("⚠️ No target (`y`) found in session. Please run data generation or upload first.")
        return

    y = st.session_state.y
    direction = st.radio("Select Transformation Direction", ["Regression ➜ Classification", "Classification ➜ Regression"])

    if direction == "Regression ➔ Classification":
        st.subheader("📊 Convert Continuous to Categorical")
        bins = st.slider("Number of Categories", 2, 10, 3)
        strategy = st.selectbox("Binning Strategy", ["uniform", "quantile", "kmeans"])

        discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)
        y_class = discretizer.fit_transform(np.array(y).reshape(-1, 1)).astype(int).ravel()

        st.session_state.y_class = y_class
        st.success("✅ Converted regression target into categorical labels.")

        st.markdown("### 🔢 Category Distribution")
        st.dataframe(pd.Series(y_class, name="y_class").value_counts().sort_index())

    elif direction == "Classification ➔ Regression":
        st.subheader("🔁 Convert Categorical to Numeric")

        if not pd.api.types.is_numeric_dtype(y):
            try:
                le = LabelEncoder()
                y_reg = le.fit_transform(y)
                st.session_state.y_reg = y_reg
                st.success("✅ Label-encoded classification target into numeric format.")

                st.markdown("### 🔢 Mapped Values")
                mapping = {label: idx for idx, label in enumerate(le.classes_)}
                st.json(mapping)

            except Exception as e:
                st.error(f"❌ Conversion failed: {e}")
        else:
            st.info("ℹ️ Target is already numeric. No conversion needed.")
            st.session_state.y_reg = y

    # Show current y summary
    st.markdown("---")
    st.markdown("### 📌 Current Target Preview")
    st.dataframe(pd.Series(st.session_state.y).head(10))
