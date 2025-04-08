# user_manual.py
import streamlit as st

def run_user_manual(deep=False, compact=False):
    st.title("📘 DAIVID Analytics User Manual")

    spacing = "" if compact else "\n\n"

    def section(title, content):
        st.markdown(f"### {title}")
        st.markdown(content + spacing)

    section("🧭 Overview", """
    DAIVID is your end-to-end decision intelligence platform. This manual provides quick explanations of each module,
    including methods, use cases, and why they matter.
    """)

    section("D: Data Exploration", f"""
    - **Notebook Scout**: Scans top Kaggle notebooks to extract trends and techniques.
    - **Auto EDA**: Automatically generates visual summaries for numeric and categorical features.
    - **Auto Feature Engineering**: Applies transformations (binning, encoding, interactions) automatically.
    - **LogReg + Interaction Explorer**: For linear models, this reveals hidden interaction terms.
    - **Distribution Auditor**: Detects shifts and anomalies in variable distributions.
    """)

    section("A: Algorithm Exploration", f"""
    - **Algorithm Selector**: Interactive guide to help select model types.
    - **AutoML Launcher**: Kicks off TPOT-based pipelines.
    - **AutoML Comparison**: Ranks and compares results across AutoML runs.
    - **Ensemble Builder**: Combines multiple models to boost performance.
    """)

    section("I: Interpretability", f"""
    - **SHAP Panel**: Explains model predictions globally and locally using SHAP values.
    - **SHAP Comparison**: Shows how SHAP importances differ across models.
    - **Golden Q&A (SHAP)**: Generates high-quality questions and smart answers using SHAP logic.
    - **Feature Importance Lab**: Aggregated visual feature rankings.
    - **Explainability Heatmap**: Categorical SHAP heatmaps by group.
    - **SHAP vs Permutation Delta Viewer**: Contrasts SHAP importances with traditional permutation-based scores.
    """)

    if deep:
        section("Deep Dive – Interpretability Methods", f"""
        - **SHAP (SHapley Additive Explanations)** uses game theory to distribute contribution scores across features.
        - **Permutation Importance** randomly shuffles feature values to see the drop in model performance.
        - **Delta Viewer** highlights where model logic diverges from performance-driven importance.
        """)

    section("V: Validation & Variants", f"""
    - **Threshold Optimizer**: Sweeps thresholds to optimize F1, precision, recall.
    - **DOE Panel**: Factorial Design of Experiments for sensitivity analysis.
    - **Model Diagnostics Lab**: Residual plots and error analysis tools.
    - **Residual Plot**: Visualizes predicted vs actual to spot issues.
    """)

    section("I: Iteration & Optimization", f"""
    - **Smart HPO Recommender**: Suggests hyperparameter ranges to explore.
    - **DAIVID HPO Engine / Trainer / Explorer**: Full control of iterative HPO.
    - **Smart Poly Finder**: Fits multiple polynomial degrees, returns fit vs overfit risk.
    - **LogReg Nonlinear Tricks**: Adds interaction/polynomial features to linear models.
    """)

    section("D: Docs & Deployment", f"""
    - **PDF Report**: Includes all charts, smart insights, golden Q&A, and SHAP plots.
    - **Saved Models**: Download and re-load trained models.
    - **Scorecard**: Self-assessment tracker for DAIVID maturity milestones.
    """)

    st.markdown("---")
    st.success("This user manual auto-updates. For custom branding, print-ready versions, or Markdown export, enable download in the PDF tab.")
