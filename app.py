import streamlit as st
import importlib

# -- Smart session state --
if "app_state" not in st.session_state:
    st.session_state.app_state = {
        "current_competition": "Titanic",
        "active_tab": "AutoML Launcher"
    }

# -- Grouped Modules by Category --
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

# -- Flat mapping for dynamic import --
DAIVID_TABS = {name: module for group in TITANIC_MODULE_GROUPS.values() for name in group for module in [
    name.lower().replace(" ", "_").replace("↔", "").replace("(", "").replace(")", "").replace("+", "_plus").replace("-", "_")
]}

# -- Icons for sidebar display --
TAB_ICONS = {
    "AutoML Launcher": "🚀", "Golden Q&A (SHAP)": "❓", "SHAP Panel": "🧠", "Notebook Scout": "📓",
    "Auto Feature Engineering": "🧬", "Auto EDA": "📊", "Distribution Auditor": "🧮", "Outlier Suppressor": "🧹",
    "Cat↔Reg Switcher": "🔄", "LogReg Nonlinear Tricks": "🎛️", "LogReg + Interaction Explorer": "🧠",
    "Algorithm Selector": "🎯", "AutoML Comparison": "📈", "Ensemble Builder": "🧩",
    "SHAP Comparison": "🔍", "SHAP Waterfall": "🌊", "Feature Importance Lab": "🎚️", "SHAP Summary Lab": "📉",
    "Explainability Heatmap": "🔥", "Correlation Matrix Lab": "🧩", "Threshold Optimizer": "⚖️", "Residual Plot": "📉",
    "Model Diagnostics Lab": "🩺", "Feature Drift Detector": "🌪️", "Target Drift Diagnostic": "📤",
    "AI-Generated Validation Scenarios": "🧠", "Sensitivity Explorer": "🌡️", "Synthetic Perturbation Tester": "🧪",
    "DOE Panel": "🧪", "Smart HPO Recommender": "🔍", "DAIVID HPO Engine": "⚙️", "DAIVID HPO Trainer": "🎓",
    "Zoomed HPO Explorer": "🔎", "Saved Models": "💾", "PDF Report": "📄", "DAIVID Analytics Scorecard": "🏆",
    "User Manual": "📘"
}

# -- Streamlit UI --
st.set_page_config(page_title="DAIVID Titanic App", layout="wide")
st.sidebar.title("🧭 DAIVID Titanic Navigator")
st.sidebar.caption("Kaggle AutoML – Modular Edition")

show_flat = st.sidebar.checkbox("🔀 Show All Modules (Flat List)", value=False)

if show_flat:
    all_tabs = list(DAIVID_TABS.keys())
    icons = [f"{TAB_ICONS.get(tab, '📌')} {tab}" for tab in all_tabs]
    selection = st.sidebar.radio("Select Module:", icons, index=all_tabs.index(st.session_state.app_state["active_tab"]))
    selected_tab = all_tabs[icons.index(selection)]
else:
    selected_tab = None
    for group, tabs in TITANIC_MODULE_GROUPS.items():
        with st.sidebar.expander(group, expanded=True):
            icons = [f"{TAB_ICONS.get(tab, '📌')} {tab}" for tab in tabs]
            selected = st.radio("", icons, key=group, label_visibility="collapsed")
            tab_name = tabs[icons.index(selected)]
            if selected:
                selected_tab = tab_name
                st.session_state.app_state["active_tab"] = tab_name
                break

# -- Dynamic Import + Run --
try:
    modname = DAIVID_TABS[selected_tab]
    module = importlib.import_module(modname)
    if hasattr(module, "run"):
        module.run()
    else:
        st.warning(f"⚠️ `{modname}` found but missing a `run()` function.")
except Exception as e:
    st.error(f"❌ Failed to load `{selected_tab}` → `{DAIVID_TABS.get(selected_tab)}`")
    st.exception(e)

st.markdown("---")
st.markdown("🧠 Powered by DAIVID – Dynamic AI for Insight, Validation, Interpretation & Discovery")
