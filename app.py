import os
print("Current files:", os.listdir())

import streamlit as st
import traceback

st.set_page_config(page_title="Titanic AutoML App", layout="wide")

from tpot_connector import _tpot_cache

# === SAFE IMPORT HELPERS ===
def safe_import(module_name, function_name):
    try:
        module = __import__(module_name)
        return getattr(module, function_name)
    except Exception as e:
        def fallback(*args, **kwargs):
            st.error(f"❌ {function_name.replace('run_', '').replace('_', ' ').title()} failed to load: {type(e).__name__}: {e}")
        return fallback

# === IMPORT MODULES ===
run_notebook_scout = safe_import("notebook_scout", "run_notebook_scout")
run_algorithm_selector = safe_import("algorithm_selector", "run_algorithm_selector")
run_saved_models_panel = safe_import("saved_models", "run_saved_models_panel")
run_distribution_auditor = safe_import("distribution_auditor", "run_distribution_auditor")
run_auto_eda = safe_import("auto_eda", "run_auto_eda")
run_autofe = safe_import("autofe", "run_autofe")
run_automl_launcher = safe_import("automl_launcher", "run_automl_launcher")
run_golden_qa = safe_import("golden_qa", "run_golden_qa")
run_shap_panel = safe_import("shap_interpretability", "run_shap_panel")
run_shap_comparison = safe_import("shap_comparison", "run_shap_comparison")
run_shap_waterfall = safe_import("shap_waterfall", "run_shap_waterfall")
run_automl_comparison = safe_import("automl_comparison", "run_automl_comparison")
run_ensemble_builder = safe_import("ensemble_builder", "run_ensemble_builder")
run_experiment_tracker = safe_import("experiment_tracker", "run_experiment_tracker")
run_doe_panel = safe_import("doe_panel", "run_doe_panel")
run_threshold_optimizer = safe_import("threshold_optimizer", "run_threshold_optimizer")
run_smart_hpo_recommender = safe_import("smart_hpo_recommender", "run_smart_hpo_recommender")
run_daivid_hpo_engine = safe_import("daivid_hpo_engine", "run_daivid_hpo_engine")
run_daivid_hpo_trainer = safe_import("daivid_hpo_trainer", "run_daivid_hpo_trainer")
run_zoom_hpo_explorer = safe_import("zoom_hpo_explorer", "run_zoom_hpo_explorer")
run_pdf_report = safe_import("pdf_report", "run_pdf_report")

# === NAVIGATION GROUPED BY DAIVID STAGES ===
st.sidebar.title("🧭 DAIVID Navigator")
phase = st.sidebar.selectbox("🔢 Phase", [
    "D: Data Exploration",
    "A: Algorithm Exploration",
    "I: Interpretability & Insights",
    "V: Validation & Variants",
    "I: Iteration & Optimization",
    "D: Documentation & Deployment"
])

phase_tabs = {
    "D: Data Exploration": ["Notebook Scout", "Auto EDA", "Auto Feature Engineering", "Distribution Auditor"],
    "A: Algorithm Exploration": ["Algorithm Selector", "AutoML Launcher", "AutoML Comparison", "Ensemble Builder"],
    "I: Interpretability & Insights": ["SHAP Panel", "SHAP Comparison", "SHAP Waterfall", "Golden Q&A"],
    "V: Validation & Variants": ["Threshold Optimizer", "DOE Panel", "Experiment Tracker"],
    "I: Iteration & Optimization": ["Smart HPO Recommender", "DAIVID HPO Engine", "DAIVID HPO Trainer", "Zoomed HPO Explorer"],
    "D: Documentation & Deployment": ["Saved Models", "PDF Report"]
}

subtab = st.sidebar.radio("🧩 Select Tab", phase_tabs[phase])

# === TAB ROUTING ===
if subtab == "Notebook Scout":
    run_notebook_scout()
elif subtab == "Algorithm Selector":
    run_algorithm_selector()
elif subtab == "Saved Models":
    run_saved_models_panel()
elif subtab == "Distribution Auditor":
    run_distribution_auditor()
elif subtab == "Auto EDA":
    run_auto_eda()
elif subtab == "Auto Feature Engineering":
    run_autofe()
elif subtab == "AutoML Launcher":
    run_automl_launcher()
elif subtab == "Golden Q&A":
    run_golden_qa()
elif subtab == "SHAP Panel":
    run_shap_panel()
elif subtab == "SHAP Comparison":
    run_shap_comparison()
elif subtab == "SHAP Waterfall":
    run_shap_waterfall()
elif subtab == "AutoML Comparison":
    run_automl_comparison()
elif subtab == "Ensemble Builder":
    run_ensemble_builder()
elif subtab == "Experiment Tracker":
    run_experiment_tracker()
elif subtab == "DOE Panel":
    if "X_train" in st.session_state and "model" in st.session_state:
        run_doe_panel(df=st.session_state["X_train"], model=st.session_state["model"])
    else:
        st.warning("🚧 Required objects missing. Train a model first to use the DOE panel.")
elif subtab == "Threshold Optimizer":
    y_true = _tpot_cache.get("y_test")
    y_proba = _tpot_cache.get("y_pred_proba")
    if y_true is not None and y_proba is not None:
        run_threshold_optimizer(y_true=y_true, y_proba=y_proba)
    else:
        st.warning("🟡 TPOT predictions not found. Please run AutoML first.")
elif subtab == "Smart HPO Recommender":
    run_smart_hpo_recommender()
elif subtab == "DAIVID HPO Engine":
    run_daivid_hpo_engine()
elif subtab == "DAIVID HPO Trainer":
    run_daivid_hpo_trainer()
elif subtab == "Zoomed HPO Explorer":
    run_zoom_hpo_explorer()
elif subtab == "PDF Report":
    run_pdf_report()
