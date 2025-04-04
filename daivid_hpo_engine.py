# smart_hpo_recommender.py
import streamlit as st
import pandas as pd
from tpot_connector import _tpot_cache


def run_smart_hpo_recommender():
    st.title("🧠 Smart Algorithm Recommender + HPO Launcher")
    st.markdown("""
    This panel recommends optimal algorithms based on your dataset characteristics and enables smart hyperparameter sweeps
    with support for feature interactions, regularization, calibrated learners, and early stopping (where supported).
    """)

    df = _tpot_cache.get("X_train")
    y = _tpot_cache.get("y_train")

    if df is None or y is None:
        st.warning("⚠️ No training data found. Please run AutoML first.")
        return

    st.markdown("### 📋 Dataset Snapshot")
    st.dataframe(df.head())

    # Basic Profiling
    st.markdown("### 🧬 Dataset Diagnostics")
    st.write(f"Shape: {df.shape}")
    st.write(f"Missing Values: {df.isnull().sum().sum()}")
    st.write(f"Numeric Features: {df.select_dtypes(include='number').shape[1]}")
    st.write(f"Categorical Features: {df.select_dtypes(exclude='number').shape[1]}")

    # AI Recommends This
    st.markdown("### 🤖 AI Recommends These Algorithms")
    recs = []
    if df.shape[0] > 2000:
        recs.append("XGBoost")
    if df.select_dtypes(include='number').shape[1] < 10:
        recs.append("Logistic Regression")
    if df.select_dtypes(exclude='number').shape[1] > 3:
        recs.append("CatBoost")
    if df.shape[1] > 15:
        recs.append("Random Forest")
    if df.select_dtypes(include='number').shape[1] > 20:
        recs.append("Neural Network")

    st.success(
        "Top Algorithm Suggestions: " + ", ".join(recs) if recs else "Unable to determine best models — run EDA first."
    )

    st.markdown("---")
    st.markdown("### 🛠️ HPO Configuration")

    model_choice = st.selectbox("Choose Model to Tune:", recs or ["Random Forest", "XGBoost", "Logistic Regression", "Neural Network"])

    st.markdown("**Early Stopping** (if supported)")
    use_early_stopping = st.checkbox("Enable Early Stopping", value=True)

    st.markdown("**Include Calibrated Learner**")
    calibrate = st.checkbox("Yes, calibrate this model")

    st.markdown("**Feature Engineering Options**")
    test_interactions = st.checkbox("Test 2- and 3-way Interactions", value=False)
    test_feature_counts = st.slider("Max # Features to Test", 3, df.shape[1], min(10, df.shape[1]))

    st.markdown("**Data Preprocessing Options**")
    norm = st.selectbox("Normalization Method:", ["None", "MinMax", "Z-Score", "Robust"])
    encoding = st.selectbox("Categorical Encoding:", ["OneHot", "Ordinal", "Binary"])
    bin_method = st.selectbox("Binning Method:", ["None", "Quantile", "Uniform", "Entropy"])
    bin_count = st.slider("# of Bins", 2, 10, 4)

    st.markdown("**Training Split**")
    test_size = st.slider("Test Size %", 10, 50, 20)

    st.markdown("**HPO Budget**")
    max_models = st.slider("Max Models to Test", 5, 50, 15)
    parallel_mode = st.checkbox("🔁 Run in Parallel (Dask mode)", value=True)

    if st.button("🚀 Launch Smart HPO"):
        st.success(f"Running {model_choice} with HPO on up to {max_models} models...")
        st.code("[Simulated launch — future: connect to HPO engine w/ grid, random, or Bayesian sweeps]")

        # Cache config if needed elsewhere
        _tpot_cache["last_hpo_config"] = {
            "model": model_choice,
            "early_stopping": use_early_stopping,
            "calibrated": calibrate,
            "interactions": test_interactions,
            "max_features": test_feature_counts,
            "norm": norm,
            "encoding": encoding,
            "bins": (bin_method, bin_count),
            "test_size": test_size,
            "max_models": max_models,
            "parallel": parallel_mode
        }

    # Smart Trail Marker
    st.markdown("---")
    st.markdown("### 🧭 What's Next?")
    st.info("""
✅ You've configured your Smart HPO settings.

➡️ **Next Step: Launch the DAIVID HPO Engine**
This will train your selected model and unlock Threshold Optimization, SHAP, and Golden Q&A.
    """)
    if st.button("🚀 Go to DAIVID HPO Engine"):
        st.session_state.tab = "DAIVID HPO Engine"
