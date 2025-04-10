import os
import ast
import sys

# Matches DAIVID_TABS logic from app.py
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

# Recreate dynamic name-to-module mapping
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

missing_files = []
no_run_func = []
syntax_errors = []

def check_run_function(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            node = ast.parse(f.read(), filename=path)
            has_run = any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in node.body)
            return has_run
    except SyntaxError as e:
        syntax_errors.append(f"{path} → ❌ SyntaxError: {e}")
        return False

for display_name, module_name in DAIVID_TABS.items():
    file_path = f"{module_name}.py"
    if not os.path.exists(file_path):
        missing_files.append(file_path)
    else:
        if not check_run_function(file_path):
            no_run_func.append(file_path)

# === Output ===
print("🧪 DAIVID Tab Module Validator\n")

if missing_files:
    print("❌ Missing Python files:")
    for f in missing_files:
        print(f" - {f}")

if no_run_func:
    print("\n⚠️ Missing `run()` function in:")
    for f in no_run_func:
        print(f" - {f}")

if syntax_errors:
    print("\n🚨 Syntax errors:")
    for err in syntax_errors:
        print(f" - {err}")

if not any([missing_files, no_run_func, syntax_errors]):
    print("✅ All modules are present and define a valid `run()` function.")

# Exit with error code if anything failed
if missing_files or no_run_func or syntax_errors:
    sys.exit(1)
