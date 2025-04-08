import streamlit as st
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from tpot_connector import latest_tpot_model, latest_X_test
import os


def run_shap_perm_delta():
    st.header("📉 SHAP vs Permutation Delta Viewer")

    model = latest_tpot_model
    X = latest_X_test

    if model is None or X is None:
        st.error("❌ Model or test data not found. Please train TPOT first.")
        return

    with st.spinner("Calculating importances..."):
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)
        shap_importance = np.abs(shap_values.values).mean(axis=0)

        perm_result = permutation_importance(model, X, model.predict(X), n_repeats=10, random_state=42)
        perm_importance = perm_result.importances_mean

        features = X.columns
        delta = shap_importance - perm_importance

        delta_df = pd.DataFrame({
            "Feature": features,
            "SHAP Importance": shap_importance,
            "Permutation Importance": perm_importance,
            "Delta": delta
        }).sort_values(by="Delta", key=abs, ascending=False)

    st.dataframe(delta_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, len(features) * 0.3))
    delta_df.plot.barh(x="Feature", y="Delta", ax=ax, color="coral")
    ax.set_title("Delta: SHAP - Permutation Importance")
    ax.axvline(0, color='gray', linestyle='--')
    st.pyplot(fig)

    # Save for PDF
    plot_path = "shap_perm_delta_plot.png"
    fig.savefig(plot_path, bbox_inches='tight')
    st.session_state["shap_perm_delta_plot_path"] = plot_path

    # Interpretation
    st.markdown("""
    #### 🤖 Interpretation
    - Large positive delta: model internally emphasizes this feature more than its real predictive power.
    - Large negative delta: this feature helps model performance but may be underweighted by internal logic.
    - Use this chart to detect leakage, multicollinearity, or surprising patterns.
    """)
