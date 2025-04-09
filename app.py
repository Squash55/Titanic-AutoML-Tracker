import os
import streamlit as st
import traceback

st.set_page_config(page_title="Titanic AutoML App", layout="wide")

from tpot_connector import _tpot_cache
from golden_qna_shap import run_golden_qna_shap
from residual_plot_panel import run_residual_plot_panel
from user_manual import run_user_manual

# === SAFE IMPORT HELPER ===
def safe_import(module_name, function_name):
    try:
        module = __import__(module_name)
        return getattr(module, function_name)
    except Exception as e:
        def fallback(*args, **kwargs):
            st.error(
                f"❌ `{function_name}` failed to load due to: **{type(e).__name__}**\n\n"
                f"**Reason:** `{e}`\n\n"
                f"📦 Check if `{module_name}.py` exists and defines `{function_name}()` properly."
            )
        return fallback

# === MODULE IMPORTS VIA SAFE WRAPPER ===
run_feature_impact_tester = safe_import("feature_impact_tester", "run_feature_impact_tester")
run_synthetic_perturbation_tester = safe_import("synthetic_perturbation_tester", "run_synthetic_perturbation_tester")
run_sensitivity_explorer = safe_import("sensitivity_explorer", "run_sensitivity_explorer")
run_model_diagnostics_lab = safe_import("model_diagnostics_lab", "run_model_diagnostics_lab")
run_ai_validation_scenarios = safe_import("ai_validation_scenarios", "run_ai_validation_scenarios")
run_target_drift_diagnostic = safe_import("target_drift_diagnostic", "run_target_drift_diagnostic")
run_feature_drift_detector = safe_import("feature_drift_detector", "run_feature_drift_detector")
run_user_manual = safe_import("user_manual", "run_user_manual")
run_shap_perm_delta = safe_import("shap_permutation_delta", "run_shap_permutation_delta")
run_explainable_boosting_visualizer = safe_import("explainable_boosting_visualizer", "run_explainable_boosting_visualizer")
run_explainable_boosting = safe_import("explainable_boosting_lab", "run_explainable_boosting")
run_smart_poly_finder = safe_import("smart_poly_finder", "run_smart_poly_finder")
run_logreg_nonlinear_lab = safe_import("logreg_nonlinear_lab", "run_logreg_nonlinear_lab")
run_logreg_nonlinear_tricks = safe_import("logreg_nonlinear_tricks", "run_logreg_nonlinear_tricks")
run_catreg_switcher = safe_import("catreg_switcher", "run_catreg_switcher")
run_synthetic_data_toggle = safe_import("synthetic_data_toggle", "run_synthetic_data_toggle")
run_daivid_roadmap = safe_import("daivid_roadmap", "run_daivid_roadmap")
run_notebook_scout = safe_import("notebook_scout", "run_notebook_scout")
run_algorithm_selector = safe_import("algorithm_selector", "run_algorithm_selector")
run_saved_models_panel = safe_import("saved_models", "run_saved_models_panel")
run_distribution_auditor = safe_import("distribution_auditor", "run_distribution_auditor")
run_auto_eda = safe_import("auto_eda", "run_auto_eda")
run_autofe = safe_import("autofe", "run_autofe")
run_outlier_suppressor = safe_import("outlier_suppressor", "run_outlier_suppressor")
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
run_logreg_interactions_explorer = safe_import("auto_fe_logreg_lab", "run_logreg_interactions_explorer")
run_feature_importance_lab = safe_import("feature_importance_lab", "run_feature_importance_lab")
run_shap_summary_lab = safe_import("shap_summary_lab", "run_shap_summary_lab")
run_shap_explainability_heatmap = safe_import("shap_explainability_heatmap", "run_shap_explainability_heatmap")
run_correlation_matrix_lab = safe_import("correlation_matrix_lab", "run_correlation_matrix_lab")
run_model_diagnostics_lab = safe_import("model_diagnostics_lab", "run_model_diagnostics_lab")
run_residual_plot_panel = safe_import("residual_plot_panel", "run_residual_plot_panel")
run_daivid_scorecard = safe_import("daivid_scorecard", "run_daivid_scorecard")

# === SIDEBAR NAVIGATION ===
st.sidebar.title("🗽 DAIVID Navigator")
show_all = st.sidebar.checkbox("📚 Show All Tabs", value=True)

# === Phase Descriptions ===
phase_descriptions = {
    "D: Data Exploration": "🧪 Explore and preprocess your dataset. Detect outliers, understand distributions, and prep features.",
    "A: Algorithm Exploration": "🤖 Select models, run AutoML, and build ensembles to optimize predictive power.",
    "I: Interpretability & Insights": "🧠 Explain your models using SHAP, feature importance, and golden Q&A.",
    "V: Validation & Variants": "🧬 Validate robustness with diagnostics, drift checks, perturbations, and DOE panels.",
    "I: Iteration & Optimization": "🔁 Tune hyperparameters, explore nonlinearities, and refine models iteratively.",
    "D: Documentation & Deployment": "📦 Save models, generate reports, and track platform maturity for deployment."
}

# === Phase Tabs ===
phase_tabs = {
    "D: Data Exploration": ["Notebook Scout", "Auto EDA", "Auto Feature Engineering", "LogReg + Interaction Explorer", "Distribution Auditor", "Outlier Suppressor", "Synthetic Data Generator", "Cat↔Reg Switcher", "LogReg Nonlinear Tricks"],
    "A: Algorithm Exploration": ["Algorithm Selector", "AutoML Launcher", "AutoML Comparison", "Ensemble Builder"],
    "I: Interpretability & Insights": ["SHAP Panel", "SHAP Comparison", "SHAP Waterfall", "Golden Q&A (SHAP)", "Golden Q&A", "Feature Importance Lab", "SHAP Summary Lab", "Explainability Heatmap", "Correlation Matrix Lab", "Explainable Boosting", "Explainable Boosting", "SHAP vs Permutation Delta Viewer", "Feature Impact Tester"],
    "V: Validation & Variants": ["Threshold Optimizer", "DOE Panel", "Experiment Tracker", "Model Diagnostics Lab", "Residual Plot", "Feature Drift Detector", "Target Drift Diagnostic", "AI-Generated Validation Scenarios", "Synthetic Perturbation Tester"],
    "I: Iteration & Optimization": ["Smart HPO Recommender", "DAIVID HPO Engine", "DAIVID HPO Trainer", "Zoomed HPO Explorer", "LogReg Nonlinear Tricks", "Smart Poly Finder", "Sensitivity Explorer"],
    "D: Documentation & Deployment": ["Saved Models", "PDF Report", "DAIVID App Maturity Scorecard"]
}

# === Sidebar Phase and Tab Selection ===
if show_all:
    st.sidebar.markdown("### 🕘️ DAIVID Roadmap")
    st.sidebar.markdown("Explore all phases and their tools:")
    for phase_name, tabs in phase_tabs.items():
        with st.sidebar.expander(f"{phase_name}", expanded=False):
            for t in tabs:
                st.markdown(f"- {t}")
    st.sidebar.markdown("---")
    phase = st.sidebar.selectbox("🔷 Choose Phase to Explore", list(phase_tabs.keys()), index=0)
else:
    phase = st.sidebar.selectbox("🔷 Phase", list(phase_tabs.keys()), index=0)

# ✅ Show description AFTER phase is selected
st.sidebar.info(phase_descriptions.get(phase, ""))

# ✅ Tab selection AFTER phase
subtab = st.sidebar.radio("🦩 Select Tab", phase_tabs[phase])

# === Route to tab functions based on selection ===
if subtab == "Notebook Scout":
    run_notebook_scout()
elif subtab == "Auto EDA":
    run_auto_eda()
elif subtab == "Auto Feature Engineering":
    run_autofe()
elif subtab == "LogReg + Interaction Explorer":
    run_algorithm_selector()  # Placeholder
elif subtab == "Distribution Auditor":
    run_distribution_auditor()
elif subtab == "Algorithm Selector":
    run_algorithm_selector()
elif subtab == "AutoML Launcher":
    run_automl_launcher()
elif subtab == "AutoML Comparison":
    run_automl_comparison()
elif subtab == "Ensemble Builder":
    run_ensemble_builder()
elif subtab == "SHAP Panel":
    run_shap_panel()
elif subtab == "SHAP Comparison":
    run_shap_comparison()
elif subtab == "SHAP Waterfall":
    run_shap_waterfall()
elif subtab == "Golden Q&A":
    run_golden_qa()
elif subtab == "Threshold Optimizer":
    y_true = _tpot_cache.get("y_test")
    y_proba = _tpot_cache.get("y_pred_proba")
    if y_true is not None and y_proba is not None:
        run_threshold_optimizer(y_true=y_true, y_proba=y_proba)
    else:
        st.warning("⚠️ TPOT predictions not found.")
elif subtab == "DOE Panel":
    run_doe_panel()
elif subtab == "Experiment Tracker":
    run_experiment_tracker()
elif subtab == "Smart HPO Recommender":
    run_smart_hpo_recommender()
elif subtab == "DAIVID HPO Engine":
    run_daivid_hpo_engine()
elif subtab == "DAIVID HPO Trainer":
    run_daivid_hpo_trainer()
elif subtab == "Zoomed HPO Explorer":
    run_zoom_hpo_explorer()
elif subtab == "Saved Models":
    run_saved_models_panel()
elif subtab == "PDF Report":
    st.sidebar.markdown("### 📄 PDF Report Options")
    st.session_state["include_ebm_pdf"] = st.sidebar.checkbox("📊 Include EBM Plot", value=True)
    st.session_state["include_shap_perm_delta_pdf"] = st.sidebar.checkbox("📉 Include SHAP vs Permutation Delta Plot", value=True)
    st.session_state["include_manual_pdf"] = st.sidebar.checkbox("📘 Include User Manual Section", value=True)
    st.session_state["include_feature_impact_pdf"] = st.sidebar.checkbox(
        "🧪 Include Feature Impact Tester Section", value=True)
    run_pdf_report()
elif subtab == "DAIVID App Maturity Scorecard":
    run_daivid_scorecard()
elif subtab == "Golden Q&A (SHAP)":
    run_golden_qna_shap()
elif subtab == "Residual Plot":
    run_residual_plot_panel()
elif subtab == "Synthetic Data Generator":
    run_synthetic_data_toggle()
elif subtab == "Outlier Suppressor":
    run_outlier_suppressor()
elif subtab == "Cat↔Reg Switcher":
    run_catreg_switcher()
elif subtab == "LogReg Nonlinear Tricks":
    run_logreg_nonlinear_tricks()
elif subtab == "Smart Poly Finder":
    run_smart_poly_finder()
elif subtab == "Explainable Boosting":
    run_explainable_boosting_visualizer()
elif subtab == "SHAP vs Permutation Delta Viewer":
    run_shap_perm_delta()
elif subtab == "DAIVID Analytics User Manual":
    run_user_manual()
elif subtab == "Feature Drift Detector":
    run_feature_drift_detector()
elif subtab == "Target Drift Diagnostic":
    run_target_drift_diagnostic()
elif subtab == "AI-Generated Validation Scenarios":
    run_ai_validation_scenarios()
elif subtab == "Model Diagnostics Lab":
    run_model_diagnostics_lab()
elif subtab == "Sensitivity Explorer":
    run_sensitivity_explorer()
elif subtab == "Synthetic Perturbation Tester":
    run_synthetic_perturbation_tester()
elif subtab == "Feature Impact Tester":
    run_feature_impact_tester()

