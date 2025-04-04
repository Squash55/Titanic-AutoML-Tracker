# shap_comparison.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from tpot_connector import _tpot_cache


def run_shap_comparison():
    st.title("🧠 SHAP Comparison Panel")
    st.markdown("""
    Compare SHAP value distributions across your top models to evaluate interpretability,
    consistency of important features, and where models diverge in influence.
    """)

    X_train = _tpot_cache.get("X_train")
    models = _tpot_cache.get("all_models", {})  # Expecting a dict of {name: model}

    if not X_train or not models:
        st.warning("⚠️ SHAP Comparison requires multiple trained models and X_train. Run AutoML first.")
        return

    top_features = set()
    shap_dfs = {}
    feature_ranks = {}
    feature_agreement = {}

    for name, model in models.items():
        try:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_train)
            shap_df = pd.DataFrame(np.abs(shap_values.values), columns=X_train.columns)
            mean_shap = shap_df.mean().sort_values(ascending=False)
            shap_dfs[name] = mean_shap
            top = mean_shap.head(5).index.tolist()
            top_features.update(top)
            feature_ranks[name] = top
        except Exception as e:
            st.error(f"SHAP failed for {name}: {e}")

    # 📈 SHAP Summary Plots
    st.markdown("### 📈 Mean Absolute SHAP by Model")
    for name, shap_series in shap_dfs.items():
        fig, ax = plt.subplots()
        shap_series.head(10).plot(kind='bar', ax=ax)
        ax.set_title(f"{name} - Top SHAP Feature Importances")
        st.pyplot(fig)

    # 🧠 Smart Summary Answers
    st.markdown("### 🧠 SHAP Smart Summary")
    if feature_ranks:
        # Most interpretable: model with highest % SHAP in top 3
        interpretable_scores = {
            name: shap_dfs[name].head(3).sum() / shap_dfs[name].sum()
            for name in shap_dfs
        }
        most_interpretable = max(interpretable_scores, key=interpretable_scores.get)

        # Consistent features: intersection of top-5s
        consistent = set.intersection(*[set(top5) for top5 in feature_ranks.values()]) if len(feature_ranks) > 1 else set()

        # Disagreements: features that appear in one but not others
        all_top = pd.Series([f for ranks in feature_ranks.values() for f in ranks])
        disagreement = all_top.value_counts()[all_top.value_counts() == 1].index.tolist()

        st.success(f"**Most Interpretable Model:** {most_interpretable} — top 3 features explain {interpretable_scores[most_interpretable]*100:.1f}% of total importance.")
        st.info(f"**Consistent Features Across Models:** {', '.join(consistent) if consistent else 'None'}")
        st.warning(f"**Disagreements in Feature Influence:** {', '.join(disagreement) if disagreement else 'None'}")
    else:
        st.info("Run multiple models to generate SHAP smart answers.")
