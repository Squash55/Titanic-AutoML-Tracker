# smart_hpo_recommender.py
import streamlit as st
import pandas as pd
from scipy.stats import normaltest
from sklearn.preprocessing import PowerTransformer
from tpot_connector import _tpot_cache

def run_smart_hpo_recommender():
    st.title("\U0001F9E0 Smart Algorithm Recommender + HPO Launcher")
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

    # Auto Normality Test
    st.markdown("### 📈 Feature Normality Check & Power Transformation")
    normality_results = []
    transformed_features = []
    all_numeric_cols = df.select_dtypes(include='number').columns.tolist()

    manual_selection = st.multiselect("Select features to power transform (overrides auto-detect):", all_numeric_cols)

    for col in all_numeric_cols:
        stat, p = normaltest(df[col].dropna())
        if p < 0.05 or col in manual_selection:
            st.warning(f"{col}: ❌ Not Normally Distributed (p = {p:.4f}) — Power Transform Recommended")
            transformed_features.append(col)
        else:
            st.success(f"{col}: ✅ Normal Distribution (p = {p:.4f})")

    if transformed_features:
        st.markdown("### ⚙️ Auto Power Transformation Preview")
        preview_cols = transformed_features[:5]
        try:
            pt = PowerTransformer(method="yeo-johnson")
            preview_data = pt.fit_transform(df[preview_cols])
            st.dataframe(pd.DataFrame(preview_data, columns=preview_cols).head())
            st.info("Applied Yeo-Johnson transformation to most non-normal features.")
        except Exception as e:
            st.error(f"Transformation failed: {e}")

    st.markdown("""
ℹ️ Features failing the normality test may benefit from power transformations (e.g., Box-Cox or Yeo-Johnson),
which can be applied automatically in the preprocessing pipeline. You may override this detection manually above.
    """)

    # AI Recommends This
    st.markdown("### \U0001F916 AI Recommends These Algorithms")
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
    st.markdown("### \U0001F6E0️ HPO Configuration")

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

    st.markdown("**PCA Dimensionality Reduction** (Optional)")
    use_pca = st.checkbox("Apply PCA to Reduce Dimensions")
    pca_components = None
    if use_pca:
        pca_mode = st.radio("Choose PCA Strategy:", ["Keep 95% Variance", "Set # Components"], horizontal=True)
        if pca_mode == "Set # Components":
            pca_components = st.slider("# PCA Components", 2, min(50, df.shape[1]), 10)
        st.info("""
✅ PCA can help if:
- You have hundreds or thousands of features
- Many features are strongly correlated
- You want to simplify a model without sacrificing too much performance

⚠️ PCA may hurt if:
- Your features are already well selected
- You need feature interpretability
- You are using tree-based models that handle collinearity well
        """)

    st.markdown("**Training Split**")
    test_size = st.slider("Test Size %", 10, 50, 20)

    st.markdown("**HPO Budget**")
    max_models = st.slider("Max Models to Test", 5, 50, 15)
    parallel_mode = st.checkbox("🔁 Run in Parallel (Dask mode)", value=True)

    st.markdown("**Optional VC Dimension Constraint**")
    vc_dim = st.slider("Max VC Dimension (Complexity Limit)", 5, 100, 30)
    st.caption("The VC (Vapnik–Chervonenkis) dimension is a theoretical upper bound on a model's complexity and generalization capacity.")

    if st.button("\U0001F680 Launch Smart HPO"):
        st.success(f"Running {model_choice} with HPO on up to {max_models} models...")
        st.code("[Simulated launch — future: connect to HPO engine w/ grid, random, or Bayesian sweeps]")

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
            "parallel": parallel_mode,
            "pca": use_pca,
            "pca_components": pca_components,
            "vc_dim": vc_dim,
            "power_transformed_features": transformed_features
        }

    # Smart Trail Marker
    st.markdown("---")
    st.markdown("### \U0001F9ED What's Next?")
    st.info(f"""
✅ You've configured your Smart HPO settings.

➡️ **Next Step: Launch the DAIVID HPO Engine**
This will train your selected model and unlock Threshold Optimization, SHAP, and Golden Q&A.

🧠 Current VC limit: {vc_dim} — Expect simpler models if set low. Reasonable target: 30–50 for balanced generalization vs flexibility.
    """)

    if st.button("🚀 Go to DAIVID HPO Trainer"):
        st.session_state.tab = "DAIVID HPO Trainer"
