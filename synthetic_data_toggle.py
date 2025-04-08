# synthetic_data_toggle.py
import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def generate_synthetic_regression_data(rows=100, seed=42):
    np.random.seed(seed)
    X = pd.DataFrame({
        'Feature_A': np.random.normal(50, 10, rows),
        'Feature_B': np.random.uniform(0, 100, rows),
        'Feature_C': np.random.randint(1, 5, rows),
    })
    y = 0.5 * X['Feature_A'] + 0.3 * X['Feature_B'] + 5 * X['Feature_C'] + np.random.normal(0, 5, rows)
    return X, y

def run_synthetic_data_toggle():
    st.title("🧪 Synthetic Data Generator")

    use_synthetic = st.toggle("Generate Synthetic Data Instead of Uploading", value=True)

    if use_synthetic:
        rows = st.slider("Number of Rows", 50, 1000, 100)
        seed = st.number_input("Random Seed", value=42)
        X, y = generate_synthetic_regression_data(rows=rows, seed=seed)

        # ✅ Store for downstream modules
        st.session_state.X = X
        st.session_state.y = y

        st.success("✅ Synthetic dataset generated!")
        st.dataframe(pd.concat([X, pd.Series(y, name='Target')], axis=1))
    else:
        uploaded = st.file_uploader("Upload your CSV dataset")
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.X = df.drop(columns=[df.columns[-1]])
            st.session_state.y = df[df.columns[-1]]
            st.success("✅ Uploaded dataset stored in session_state!")
            st.dataframe(df.head())

    # Auto route to Cat ↔ Reg converter if user wants
    if st.checkbox("🔁 Open Cat ↔ Reg Switcher"):
        from catreg_switcher import run_catreg_switcher
        run_catreg_switcher()
