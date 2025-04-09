# user_manual.py
import streamlit as st

# === Function to generate downloadable Markdown content ===
def generate_manual_markdown(include_images=True):
    sections = []

    sections.append("## 📊 Auto EDA\nExploratory Data Analysis that auto-generates charts and patterns. Useful for spotting outliers, missing values, and variable distributions.")
    if include_images:
        sections.append("![Auto EDA Example](https://example.com/eda.png)")  # Replace with actual image paths

    sections.append("## 🧠 Golden Q&A\nAI-powered insights using SHAP and question templates to explain patterns.")
    if include_images:
        sections.append("![Golden QA Example](https://example.com/goldenqa.png)")

    return "\n\n".join(sections)

# === MAIN MANUAL RUNNER ===
def run_user_manual(deep=True, compact=True):
    st.title("📘 DAIVID Analytics User Manual")

    # === 📚 Glossary Sidebar ===
    with st.sidebar.expander("📚 Glossary", expanded=False):
        st.markdown("""
        - **EDA**: Exploratory Data Analysis – the process of visually and statistically understanding a dataset.
        - **SHAP**: SHapley Additive exPlanations – explains each prediction by computing the contribution of every feature.
        - **HPO**: Hyperparameter Optimization – tuning settings of algorithms to maximize model performance.
        - **PDF Report**: A downloadable document summarizing model, insights, and visuals.
        """)

    # === Sidebar Export Options ===
    with st.sidebar.expander("📘 Export Options", expanded=False):
        st.session_state["manual_image_mode"] = st.checkbox("🖼️ Include Visual Aids", value=True)
        if st.button("📥 Prepare Markdown Download"):
            st.download_button(
                label="📄 Download Manual (.md)",
                data=generate_manual_markdown(include_images=st.session_state["manual_image_mode"]),
                file_name="DAIVID_Analytics_User_Manual.md",
                mime="text/markdown"
            )

    spacing = "" if compact else "\n\n"

    def section(title, content):
        st.markdown(f"### {title}")
        st.markdown(content + spacing)

    # === Manual Content ===
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
    - **Feature Drift Detector**: Compares training vs new data to identify features with changed distributions. Uses:
    - KS Test for numeric features (sensitive to shape shifts),
    - Chi² Test for categorical features (detects category imbalance).
    - Useful for alerting model owners when assumptions no longer hold.
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
