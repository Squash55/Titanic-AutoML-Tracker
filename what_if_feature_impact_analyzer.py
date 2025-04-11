import streamlit as st
import pandas as pd
import numpy as np
from tpot_connector import _tpot_cache

def run_what_if_feature_impact_analyzer():
    st.title("🔍 What-If Feature Impact Analyzer")

    # Stated Purpose
    st.markdown("""
    This tool analyzes the impact of modifying or removing individual features on model predictions.
    It helps you understand how each feature contributes to the model's decisions and identifies which features are fragile, dominant, or negligible.
    """)

    model = _tpot_cache.get("latest_tpot_model")
    X_train = _tpot_cache.get("latest_X_train")

    if model is None or X_train is None:
        st.warning("⚠️ TPOT model or training data missing. Please run AutoML first.")
        return

    # Instance selection for analysis
    instance_idx = st.slider("🔢 Choose a training row to analyze", 0, len(X_train) - 1, 0)
    instance = X_train.iloc[[instance_idx]]
    st.dataframe(instance)

    # Get baseline prediction
    try:
        baseline_pred = model.predict(instance)[0]
        proba_available = hasattr(model, "predict_proba")
        if proba_available:
            baseline_proba = model.predict_proba(instance)[0]
    except Exception as e:
        st.error(f"❌ Prediction failed: {type(e).__name__}: {e}")
        return

    # Impact test loop
    results = []
    for col in X_train.columns:
        modified = instance.copy()

        if pd.api.types.is_numeric_dtype(modified[col]):
            modified[col] = 0
        else:
            modified[col] = None

        try:
            new_pred = model.predict(modified)[0]
            if proba_available:
                new_proba = model.predict_proba(modified)[0]
                delta = np.abs(baseline_proba - new_proba).sum()
            else:
                delta = float(baseline_pred != new_pred)

            results.append({
                "Feature": col,
                "Changed Prediction": new_pred,
                "Impact Score": round(delta, 4)
            })
        except Exception as e:
            results.append({
                "Feature": col,
                "Changed Prediction": "Error",
                "Impact Score": None
            })

    df = pd.DataFrame(results).sort_values(by="Impact Score", ascending=False)
    st.markdown("### 📊 Feature Impact Summary")
    st.dataframe(df, use_container_width=True)

    # Download CSV with impact results
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Impact Report", data=csv, file_name="feature_impact_report.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("""
    ### 🧠 Interpretation
    - **High impact scores**: These features drastically affect model predictions. Consider focusing on these for optimization, validation, or explainability.
    - **Low or zero impact**: These features may be robust or irrelevant, and could potentially be pruned from the model.
    - **Use in What-If analysis**: This tool is ideal for feature pruning, explainability, or testing adversarial robustness by modifying specific features and observing changes in the model's behavior.
    """)
