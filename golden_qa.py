# golden_qa.py

import streamlit as st
import shap
from tpot_connector import __dict__ as _tpot_cache
import numpy as np

def run_golden_qa():
    st.subheader("🧠 Golden Q&A with SHAP & Threshold Intelligence")

    model = _tpot_cache.get("latest_tpot_model") or _tpot_cache.get("latest_rf_model")
    X_train = _tpot_cache.get("latest_X_train")
    threshold = _tpot_cache.get("selected_threshold")

    if model is None or X_train is None:
        st.warning("⚠️ No model or training data found. Run TPOT or RandomForest first.")
        return

    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)
        feature_importances = np.abs(shap_values.values).mean(axis=0)
        top_features = sorted(
            zip(X_train.columns, feature_importances),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        st.markdown("### 🔍 Top Feature Drivers (via SHAP)")
        for feature, score in top_features:
            st.markdown(f"- **{feature}** → avg impact: `{score:.3f}`")

        st.markdown("### 💡 Golden Insights")
        if threshold is not None:
            st.success(f"Current classification threshold is **{threshold:.2f}**.")
            if threshold < 0.4:
                st.info("This favors **recall** over precision — you're catching more positives, but may have more false alarms.")
            elif threshold > 0.6:
                st.info("This favors **precision** — you’ll reduce false positives, but may miss some positives.")
            else:
                st.info("You're using a **balanced threshold**. Adjust only if your business case demands more precision or recall.")

        top_feature_names = [f for f, _ in top_features]
        st.markdown("### 🧭 Suggested Golden Questions")
        for f in top_feature_names:
            st.markdown(f"- Why does **{f}** have such a strong influence?")
            st.markdown(f"- Can we improve **{f}** during training or feature engineering?")
            st.markdown(f"- Is **{f}** actionable or just correlated?")

    except Exception as e:
        st.error(f"❌ SHAP analysis failed: {type(e).__name__}: {e}")
