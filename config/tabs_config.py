# config/tabs_config.py
import os  # Ensure os is imported to handle file checks

TITANIC_MODULE_GROUPS = {
    "🟢 Core (Data Prep)": [
        "Notebook Scout", "Auto EDA", "Auto Feature Engineering", "Distribution Auditor",
        "Outlier Suppressor", "Cat↔Reg Switcher", "LogReg Nonlinear Tricks", "LogReg + Interaction Explorer"
    ],
    "🟡 Mid (Modeling)": [
        "Algorithm Selector", "automl Launcher", "AutoML Comparison", "Ensemble Builder"
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
    for group in TITANIC_MODULE_GROUPS.values()
    for name in group
}

# Add the run() function to each tab file if not already created
for module_name in DAIVID_TABS.values():
    tab_file = f"tabs/{module_name}.py"
    
    # If the file doesn't exist, create it
    if not os.path.exists(tab_file):
        with open(tab_file, "w", encoding="utf-8") as f:
            f.write(f"""\"\"\"
Auto-generated tab: {module_name}
\"\"\"

import streamlit as st

@st.cache_data
def run():
    st.title("{module_name}")
    st.info("This is the auto-generated tab for {module_name}. Add your custom content here.")
""")
        print(f"✅ Created: {tab_file}")
    else:
        print(f"⚠️ File {tab_file} already exists.")
