# automl_comparison.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tpot_connector import _tpot_cache


def run_automl_comparison():
    st.title("📊 AutoML Comparison Dashboard")
    st.markdown("""
    This panel compares the performance of all AutoML models you've run.
    Metrics include accuracy, F1, AUC, feature count, and training time.
    Use this to identify the best-performing model or spot overfitting.
    """)

    # Load cached results from TPOT and H2O if available
    model_results = _tpot_cache.get("model_results", pd.DataFrame())

    if model_results.empty:
        st.warning("⚠️ No model results found. Run TPOT or H2O AutoML to populate this table.")
        return

    # Metric selection
    metric_to_sort = st.selectbox("🔢 Sort models by:", ["Accuracy", "F1", "AUC", "Training Time", "Delta (Overfit)"])
    ascending_sort = metric_to_sort == "Training Time"  # For time, lower is better

    # Sort and display
    sorted_results = model_results.sort_values(by=metric_to_sort, ascending=ascending_sort)
    st.dataframe(sorted_results)

    # Highlight best performer
    best_model = sorted_results.iloc[0]
    st.success(f"🏆 Best Model: {best_model['Model Name']} (Top {metric_to_sort}: {best_model[metric_to_sort]:.4f})")

    # Bar plots
    st.markdown("### 📈 Metric Bar Plots")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=sorted_results, x="Model Name", y=metric_to_sort, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title(f"Model Comparison by {metric_to_sort}")
    st.pyplot(fig)

    # Show overfitting spread
    st.markdown("### ⚠️ Train vs Test Delta (Overfitting Risk)")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=sorted_results, x="Model Name", y="Delta (Overfit)", ax=ax2)
    ax2.axhline(0.1, color='red', linestyle='--', label='Risk Threshold')
    ax2.legend()
    ax2.set_title("Train-Test Gap per Model")
    st.pyplot(fig2)

    # SHAP Hook Placeholder
    st.markdown("### 🧠 SHAP Comparison (Coming Soon)")
    st.info("Side-by-side SHAP importance plots will appear here once multiple models are available.")
