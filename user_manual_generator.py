# user_manual_generator.py
import streamlit as st
import pandas as pd
import datetime
from fpdf import FPDF

# === TAB DESCRIPTIONS ===
TAB_DESCRIPTIONS = {
    "Notebook Scout": "Scans and summarizes top Kaggle notebooks for insights, modeling strategies, and feature ideas.",
    "Auto EDA": "Performs automated exploratory data analysis on your dataset, highlighting key stats, distributions, and patterns.",
    "Auto Feature Engineering": "Suggests and applies automated feature engineering techniques to improve model performance.",
    "LogReg + Interaction Explorer": "Visualizes logistic regression with interactions to reveal nonlinear feature impacts.",
    "Distribution Auditor": "Audits the distribution of features to flag skewness, imbalance, or outliers.",
    "Algorithm Selector": "Presents an interactive catalog of ML algorithms with pros/cons and use cases.",
    "AutoML Launcher": "Runs automated model training using tools like TPOT and H2O.",
    "AutoML Comparison": "Compares multiple AutoML model results in a unified view.",
    "Ensemble Builder": "Helps create stacked models or voting ensembles from top performers.",
    "SHAP Panel": "Visualizes SHAP explanations to understand feature influence.",
    "SHAP Comparison": "Compares SHAP outputs across multiple models side-by-side.",
    "SHAP Waterfall": "Waterfall plots show prediction breakdowns for individual rows.",
    "Golden Q&A": "Rule-based questions and answers extracted from the dataset.",
    "Golden Q&A (SHAP)": "AI-enhanced answers to golden questions using SHAP values.",
    "Feature Importance Lab": "Compares different methods of computing feature importance.",
    "SHAP Summary Lab": "Summarizes SHAP values into a global ranking.",
    "Explainability Heatmap": "Heatmap showing where explainability diverges across groups.",
    "Correlation Matrix Lab": "Displays correlation matrix with p-values and triangle masking.",
    "Threshold Optimizer": "Sweeps classification thresholds to optimize precision/recall tradeoff.",
    "DOE Panel": "Applies Design of Experiments (DOE) to test feature interactions.",
    "Experiment Tracker": "Logs experiment metadata, performance, and notes.",
    "Model Diagnostics Lab": "Plots residuals, learning curves, and overfit detection visuals.",
    "Residual Plot": "Displays prediction residuals vs true values.",
    "Smart Poly Finder": "Tests polynomial degrees and flags overfit risk via p-values.",
    "Smart HPO Recommender": "Suggests hyperparameters to optimize based on SHAP + search history.",
    "DAIVID HPO Engine": "Runs grid/randomized hyperparameter search.",
    "DAIVID HPO Trainer": "Trains models with optimized hyperparameters.",
    "Zoomed HPO Explorer": "Visual explorer of HPO results, performance vs params.",
    "Saved Models": "View, download, or delete previously trained models.",
    "PDF Report": "Generates downloadable summary report with plots and answers.",
    "DAIVID App Maturity Scorecard": "Evaluates app coverage and quality across key analytics capabilities.",
    "Synthetic Data Generator": "Generates synthetic regression/classification datasets.",
    "Outlier Suppressor": "Detects and optionally adjusts or removes extreme values.",
    "Cat↔Reg Switcher": "Converts categorical targets into bins or continuous targets into classes.",
    "LogReg Nonlinear Tricks": "Adds non-linear interactions or transformations to improve logistic regression.",
    "Explainable Boosting": "Runs EBM (Explainable Boosting Machine) and visualizes feature plots.",
    "SHAP vs Permutation Delta Viewer": "Compares SHAP importance vs permutation importance to detect hidden bias."
}

# === STREAMLIT LOGIC ===
def run_user_manual():
    st.title("📘 DAIVID Analytics User Manual")
    mode = st.radio("Select View Mode:", ["🧑‍💻 Developer Mode", "🎯 Customer Mode"], horizontal=True)

    if st.button("🔁 Regenerate Manual"):
        st.session_state.manual_df = pd.DataFrame([
            {"Tab": tab, "Explanation": desc}
            for tab, desc in TAB_DESCRIPTIONS.items()
            if mode == "🧑‍💻 Developer Mode" or "Golden" in tab or "SHAP" in tab or "PDF" in tab
        ])
        st.success("✅ Manual regenerated.")

    df = st.session_state.get("manual_df")
    if df is not None:
        st.dataframe(df, use_container_width=True)

        # Download buttons
        md_text = "\n".join([f"### {row['Tab']}\n{row['Explanation']}" for _, row in df.iterrows()])

        st.download_button("📥 Download Manual (.md)", data=md_text, file_name="DAIVID_Analytics_User_Manual.md")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for _, row in df.iterrows():
            pdf.set_font("Arial", 'B', 12)
            pdf.multi_cell(0, 10, row['Tab'])
            pdf.set_font("Arial", '', 11)
            pdf.multi_cell(0, 8, row['Explanation'])
            pdf.ln(3)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf.output(tmp_file.name)
            with open(tmp_file.name, "rb") as f:
                st.download_button("📥 Download Manual (.pdf)", data=f, file_name="DAIVID_Analytics_User_Manual.pdf")
