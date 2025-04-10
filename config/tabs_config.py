# config/tabs_config.py
import os  # Add this import to fix the error

TITANIC_MODULE_GROUPS = {
    "🟢 Core (Data Prep)": [
        "Notebook Scout", "Auto EDA", "Auto Feature Engineering", "Distribution Auditor",
        "Outlier Suppressor", "Cat↔Reg Switcher", "LogReg Nonlinear Tricks", "LogReg + Interaction Explorer"
    ],
    "🟡 Mid (Modeling)": [
        "Algorithm Selector", "AutoML Launcher", "AutoML Comparison", "Ensemble Builder"
    ],
    "🟣 Advanced: Interpretability": [
        "SHAP Panel", "Golden Q&A (SHAP)", "SHAP Comparison", "SHAP Waterfall",
        "Feature Importance Lab", "SHAP Summary Lab", "Explainability Heatmap", "Correlation Matrix Lab"
    ],
    "🟣 Advanced: Validation & Drift": [
        "Threshold Optimizer", "Residual Plot", "Model Diagnostics Lab",
        "Feature Drift Detector", "Target Drift Diagnostic", "AI-Generated Validation Scenarios"
    ],
    "🟣 Advanced: Stress Testing": [
        "Sensitivity Explorer", "Synthetic Perturbation Tester", "DOE Panel"
    ],
    "🟣 Advanced: Optimization": [
        "Smart HPO Recommender", "DAIVID HPO Engine", "DAIVID HPO Trainer", "Zoomed HPO Explorer"
    ],
    "📦 Deployment & Docs": [
        "Saved Models", "PDF Report", "DAIVID Analytics Scorecard", "User Manual"
    ]
}

DAIVID_TABS = {
    name: name.lower()
                .replace(" ", "_")
                .replace("↔", "")
                .replace("(", "")
                .replace(")", "")
                .replace("+", "_plus")
                .replace("-", "_")
    for group in TIT
