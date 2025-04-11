import streamlit as st 
import pandas as pd
import numpy as np
from tpot_connector import __dict__ as _tpot_cache
import shap

def run():
    st.title("📊 Auto EDA Dashboard (Safe Mode)")

    # 1. Grab cached values
    df = _tpot_cache.get("latest_X_train")
    y = _tpot_cache.get("latest_y_train")
    model = _tpot_cache.get("latest_tpot_model") or _tpot_cache.get("latest_rf_model")

    # 2. Show detailed inspection
    with st.expander("📦 Cached Data Overview"):
        st.write("🔹 `latest_X_train`:", type(df), df.shape if isinstance(df, pd.DataFrame) else None)
        st.write("🔹 `latest_y_train`:", type(y), y.shape if hasattr(y, 'shape') else None)
        st.write("🔹 `latest_model`:", type(model), str(model)[:100])

    # 3. Show alerts
    if df is None:
        st.error("❌ `latest_X_train` is missing from cache. You may need to run TPOT or RF first.")
    if y is None:
        st.error("❌ `latest_y_train` is missing from cache.")
    if model is None:
        st.warning("⚠️ No trained model (TPOT or RandomForest) found. SHAP and nomogram views may be limited.")

    # 4. Proceed only if core data exists
    if df is None or y is None:
        st.stop()

    df = df.copy()
    df["target"] = y

    st.success("✅ Auto EDA inputs are valid. Charts will be enabled after safe-mode passes.")

    # AI Insights: Show basic statistical insights about the data
    st.subheader("📊 Basic Statistical Insights")
    st.write(df.describe())

    # Feature importance (if a model exists)
    if model:
        if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            importances = model.feature_importances_
            feature_names = df.columns
            st.subheader("🔍 Feature Importance")
            st.bar_chart(dict(zip(feature_names, importances)))

        elif isinstance(model, TPOTClassifier):  # If TPOT model is available
            st.subheader("🔍 SHAP Feature Importance")
            explainer = shap.TreeExplainer(model.fitted_pipeline_)
            shap_values = explainer.shap_values(df)
            shap.summary_plot(shap_values, df)

    # Placeholder for switching tabs
    eda_tab = st.selectbox("📌 Choose a Chart to Render", [
        "Main Effects", "Pairwise Scatter", "Parallel Coordinates", "Nomogram",
        "Raincloud", "Jittered Categorical", "3D Surface"
    ])
    st.info(f"🛠 Chart rendering will appear here soon for: `{eda_tab}`")

    # Once everything is confirmed working, plug in the visuals in order
