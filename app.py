# app.py
import os
import streamlit as st
from tpot_connector import _tpot_cache

st.set_page_config(page_title="DAIVID Analytics App", layout="wide")

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

# === MODULE IMPORTS ===
run_notebook_scout = safe_import("notebook_scout", "run_notebook_scout")
run_auto_eda = safe_import("auto_eda", "run_auto_eda")
run_autofe = safe_import("autofe", "run_autofe")
run_distribution_auditor = safe_import("distribution_auditor", "run_distribution_auditor")
run_outlier_suppressor = safe_import("outlier_suppressor", "run_outlier_suppressor")
run_catreg_switcher = safe_import("catreg_switcher", "run_catreg_switcher")
run_logreg_nonlinear_tricks = safe_import("logreg_nonlinear_tricks", "run_logreg_nonlinear_tricks")
run_logreg_interactions_explorer = safe_import("auto_fe_logreg_lab", "run_logreg_interactions_explorer")

run_algorithm_selector = safe_import("algorithm_selector", "run_algorithm_selector")
run_automl_launcher = safe_import("automl_launcher", "run_automl_launcher")
run_automl_comparison = safe_import("automl_comparison", "run_automl_comparison")
run_ensemble_builder = safe_import("ensemble_builder", "run_ensemble_builder")

run_shap_panel = safe_import("shap_interpretability", "run_shap_panel")
run_golden_qna_shap = safe_import("golden_qna_shap", "run_golden_qna_shap")
run_shap_comparison = safe_import("shap_comparison", "run_shap_comparison")
run_shap_waterfall = safe_import("shap_waterfall", "run_shap_waterfall")
run_feature_importance_lab = safe_import("feature_importance_lab", "run_feature_importance_lab")
run_shap_summary_lab = safe_import("shap_summary_lab", "run_shap_summary_lab")
run_shap_explainability_heatmap = safe_import("shap_explainability_heatmap", "run_shap_explainability_heatmap")
run_correlation_matrix_lab = safe_import("correlation_matrix_lab", "run_correlation_matrix_lab")

run_threshold_optimizer = safe_import("threshold_optimizer", "run_threshold_optimizer")
run_residual_plot_panel = safe_import("residual_plot_panel", "run_residual_plot_panel")
run_model_diagnostics_lab = safe_import("model_diagnostics_lab", "run_model_diagnostics_lab")
run_feature_drift_detector = safe_import("feature_drift_detector", "run_feature_drift_detector")
run_target_drift_diagnostic = safe_import("target_drift_diagnostic", "run_target_drift_diagnostic")
run_ai_validation_scenarios = safe_import("ai_validation_scenarios", "run_ai_validation_scenarios")
run_synthetic_perturbation_tester = safe_import("synthetic_perturbation_tester", "run_synthetic_perturbation_tester")
run_doe_panel = safe_import("doe_panel", "run_doe_panel")
run_sensitivity_explorer = safe_import("sensitivity_explorer", "run_sensitivity_explorer")

run_smart_hpo_recommender = safe_import("smart_hpo_recommender", "run_smart_hpo_recommender")
run_daivid_hpo_engine = safe_import("daivid_hpo_engine", "run_daivid_hpo_engine")
run_daivid_hpo_trainer = safe_import("daivid_hpo_trainer", "run_daivid_hpo_trainer")
run_zoom_hpo_explorer = safe_import("zoom_hpo_explorer", "run_zoom_hpo_explorer")

run_pdf_report = safe_import("pdf_report", "run_pdf_report")
run_saved_models_panel = safe_import("saved_models", "run_saved_models_panel")
run_daivid_scorecard = safe_import("daivid_scorecard", "run_daivid_scorecard")
run_user_manual = safe_import("user_manual", "run_user_manual")

# === SIDEBAR NAVIGATION ===
st.sidebar.title("🗽 DAIVID Analytics Navigator")
st.sidebar.caption("Select a module to explore")

analysis_levels = {
    "🟢 Core (Data Prep)": [
        "Notebook Scout", "Auto EDA", "Auto Feature Engineering", "Distribution Auditor",
        "Outlier Suppressor", "Cat↔Reg Switcher", "LogReg Nonlinear Tricks", "LogReg + Interaction Explorer"
    ],
    "🟡 Mid (Modeling)": [
        "Algorithm Selector", "AutoML Launcher", "AutoML Comparison", "Ensemble Builder"
    ],
    "🔵 Interpretability": [
        "SHAP Panel", "Golden Q&A (SHAP)", "SHAP Comparison", "SHAP Waterfall",
        "Feature Importance Lab", "SHAP Summary Lab", "Explainability Heatmap", "Correlation Matrix Lab"
    ],
    "🟣 Validation & Drift": [
        "Threshold Optimizer", "Residual Plot", "Model Diagnostics Lab",
        "Feature Drift Detector", "Target Drift Diagnostic", "AI-Generated Validation Scenarios"
    ],
    "🧪 Stress Testing": [
        "Sensitivity Explorer", "Synthetic Perturbation Tester", "DOE Panel"
    ],
    "🔁 Optimization": [
        "Smart HPO Recommender", "DAIVID HPO Engine", "DAIVID HPO Trainer", "Zoomed HPO Explorer"
    ],
    "📦 Deployment & Docs": ["Saved Models", "PDF Report", "DAIVID Analytics Scorecard", "User Manual"]
}

level = st.sidebar.selectbox("📚 Select Level", list(analysis_levels.keys()), index=0)
module = st.sidebar.radio("🧪 Module", analysis_levels[level])

# === ROUTING ===
if module == "Notebook Scout":
    run_notebook_scout()
elif module == "Auto EDA":
    run_auto_eda()
elif module == "Auto Feature Engineering":
    run_autofe()
elif module == "Distribution Auditor":
    run_distribution_auditor()
elif module == "Outlier Suppressor":
    run_outlier_suppressor()
elif module == "Cat↔Reg Switcher":
    run_catreg_switcher()
elif module == "LogReg Nonlinear Tricks":
    run_logreg_nonlinear_tricks()
elif module == "LogReg + Interaction Explorer":
    run_logreg_interactions_explorer()

elif module == "Algorithm Selector":
    run_algorithm_selector()
elif module == "AutoML Launcher":
    run_automl_launcher()
elif module == "AutoML Comparison":
    run_automl_comparison()
elif module == "Ensemble Builder":
    run_ensemble_builder()

elif module == "SHAP Panel":
    run_shap_panel()
elif module == "Golden Q&A (SHAP)":
    run_golden_qna_shap()
elif module == "SHAP Comparison":
    run_shap_comparison()
elif module == "SHAP Waterfall":
    run_shap_waterfall()
elif module == "Feature Importance Lab":
    run_feature_importance_lab()
elif module == "SHAP Summary Lab":
    run_shap_summary_lab()
elif module == "Explainability Heatmap":
    run_shap_explainability_heatmap()
elif module == "Correlation Matrix Lab":
    run_correlation_matrix_lab()

elif module == "Threshold Optimizer":
    run_threshold_optimizer()
elif module == "Residual Plot":
    run_residual_plot_panel()
elif module == "Model Diagnostics Lab":
    run_model_diagnostics_lab()
elif module == "Feature Drift Detector":
    run_feature_drift_detector()
elif module == "Target Drift Diagnostic":
    run_target_drift_diagnostic()
elif module == "AI-Generated Validation Scenarios":
    run_ai_validation_scenarios()

elif module == "Sensitivity Explorer":
    run_sensitivity_explorer()
elif module == "Synthetic Perturbation Tester":
    run_synthetic_perturbation_tester()
elif module == "DOE Panel":
    run_doe_panel()

elif module == "Smart HPO Recommender":
    run_smart_hpo_recommender()
elif module == "DAIVID HPO Engine":
    run_daivid_hpo_engine()
elif module == "DAIVID HPO Trainer":
    run_daivid_hpo_trainer()
elif module == "Zoomed HPO Explorer":
    run_zoom_hpo_explorer()

elif module == "Saved Models":
    run_saved_models_panel()
elif module == "PDF Report":
    run_pdf_report()
elif module == "DAIVID Analytics Scorecard":
    run_daivid_scorecard()
elif module == "User Manual":
    run_user_manual()
else:
    st.warning("🚧 Module not yet connected. Check back soon!")
