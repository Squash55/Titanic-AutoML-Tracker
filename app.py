import streamlit as st
import importlib
from config.tabs_config import TITANIC_MODULE_GROUPS, DAIVID_TABS

# -- Safe session state init --
if "app_state" not in st.session_state:
    st.session_state.app_state = {
        "active_tab": "AutoML Launcher"
    }

# -- Icons for sidebar display --
TAB_ICONS = {
    "AutoML Launcher": "🚀", "Golden Q&A (SHAP)": "❓", "SHAP Panel": "🧠", "Notebook Scout": "📓",
    "Auto Feature Engineering": "🧬", "Auto EDA": "📊", "Distribution Auditor": "🧮", "Outlier Suppressor": "🧹",
    "Cat ↔ Reg Switcher": "🔄", "LogReg Nonlinear Tricks": "🎛️", "LogReg + Interaction Explorer": "🧠",
    "Algorithm Selector": "🎯", "AutoML Comparison": "📈", "Ensemble Builder": "🧩",
    "SHAP Comparison": "🔍", "SHAP Waterfall": "🌊", "Feature Importance Lab": "🎚️", "SHAP Summary Lab": "📉",
    "Explainability Heatmap": "🔥", "Correlation Matrix Lab": "🧩", "Threshold Optimizer": "⚖️", "Residual Plot": "📉",
    "Model Diagnostics Lab": "🩺", "Feature Drift Detector": "🌪️", "Target Drift Diagnostic": "📤",
    "AI-Generated Validation Scenarios": "🧠", "Sensitivity Explorer": "🌡️", "Synthetic Perturbation Tester": "🧪",
    "DOE Panel": "🧪", "Smart HPO Recommender": "🔍", "DAIVID HPO Engine": "⚙️", "DAIVID HPO Trainer": "🎓",
    "Zoomed HPO Explorer": "🔎", "Saved Models": "💾", "PDF Report": "📄", "DAIVID Analytics Scorecard": "🏆",
    "User Manual": "📘", "AutoML & AI Validation": "⚙️"  # Add this line
}


# -- Streamlit UI --
st.set_page_config(page_title="DAIVID Analytics App", layout="wide")
st.sidebar.title("🧭 DAIVID Analytics Navigator")
st.sidebar.caption("Dynamic AI for Insight, Validation, Interpretation & Discovery")

show_flat = st.sidebar.checkbox("🔀 Show All Modules (Flat List)", value=False)

if show_flat:
    all_tabs = list(DAIVID_TABS.keys())
    display_names = [f"{TAB_ICONS.get(tab, '📌')} {tab}" for tab in all_tabs]
    default_index = display_names.index(f"{TAB_ICONS.get(st.session_state.app_state['active_tab'], '📌')} {st.session_state.app_state['active_tab']}")
    selection = st.sidebar.radio("Select Module:", display_names, index=default_index)
    selected_tab = all_tabs[display_names.index(selection)]
else:
    display_names = []
    name_to_tab = {}
    for group, tab_list in TITANIC_MODULE_GROUPS.items():
        for tab in tab_list:
            label = f"{TAB_ICONS.get(tab, '📌')} [{group.split(' ')[1]}] {tab}"
            display_names.append(label)
            name_to_tab[label] = tab
    default_label = next((k for k, v in name_to_tab.items() if v == st.session_state.app_state['active_tab']), list(name_to_tab.keys())[0])
    selection = st.sidebar.radio("📚 Select Module:", display_names, index=display_names.index(default_label))
    selected_tab = name_to_tab[selection]

st.session_state.app_state["active_tab"] = selected_tab

# -- Dynamic Import + Run --
try:
    modname = DAIVID_TABS[selected_tab]

    # Manually import the 'catreg_switcher' to test if it exists
    if modname == "catreg_switcher":
        import catreg_switcher  # This is the manual import check
        st.success("✅ Successfully imported catreg_switcher manually!")
    
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
