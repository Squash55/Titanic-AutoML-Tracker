# config/tabs_config.py

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
    "AI-Generated Validation Scenarios": "ai_validation_scenarios",
    "auto_eda": "auto_eda.py",
    "auto_fe_logreg_lab": "auto_fe_logreg_lab.py",
    "auto_feature_engineering": "autofe.py",
    "auto_threshold_optimizer": "auto_threshold_optimizer.py",
    "auto_ml_comparison": "automl_comparison.py",
    "auto_ml_launcher": "automl_launcher.py",
    "catreg_switcher": "catreg_switcher.py",
    "daivid_hpo_engine": "daivid_hpo_engine.py",
    "daivid_hpo_trainer": "daivid_hpo_trainer.py",
    "daivid_roadmap": "daivid_roadmap.py",
    "daivid_scorecard": "daivid_scorecard.py",
    "distribution_auditor": "distribution_auditor.py",
    "doe_panel": "doe_panel.py",
    "ensemble_builder": "ensemble_builder.py",
    "experiment_tracker": "experiment_tracker.py",
    "feature_drift_detector": "feature_drift_detector.py",
    "feature_engineering": "feature_engineering.py",
    "feature_impact_tester": "feature_impact_tester.py",
    "feature_importance_compare": "feature_importance_compare.py",
    "generate_missing_tabs": "generate_missing_tabs.py",
    "golden_qna_shap": "golden_qna_shap.py",
    "logreg_nonlinear_lab": "logreg_nonlinear_lab.py",
    "logreg_nonlinear_tricks": "logreg_nonlinear_tricks.py",
    "model_diagnostics_lab": "model_diagnostics_lab.py",
    "model_leaderboard_panel": "model_leaderboard_panel.py",
    "notebook_insights": "notebook_insights.py",
    "notebook_scout": "notebook_scout.py",
    "outlier_suppressor": "outlier_suppressor.py",
    "pdf_report": "pdf_report.py",
    "residual_plot_panel": "residual_plot_panel.py",
    "sensitivity_explorer": "sensitivity_explorer.py",
    "shap_comparison": "shap_comparison.py",
    "shap_interpretability": "shap_interpretability.py",
    "shap_permutation_delta": "shap_permutation_delta.py",
    "shap_waterfall": "shap_waterfall.py",
    "smart_hpo_recommender": "smart_hpo_recommender.py",
    "smart_poly_finder": "smart_poly_finder.py",
    "synthetic_data_toggle": "synthetic_data_toggle.py",
    "synthetic_perturbation_tester": "synthetic_perturbation_tester.py",
    "test_imports": "test_imports.py",
    "threshold_backtester": "threshold_backtester.py",
    "threshold_optimizer": "threshold_optimizer.py",
    "titanic_sample": "titanic_sample.csv",
    "tpot_connector": "tpot_connector.py",
    "tpot_saver": "tpot_saver.py",
    "user_manual": "user_manual.py",
    "user_manual_generator": "user_manual_generator.py",
    "zoom_hpo_explorer": "zoom_hpo_explorer.py",
}

# Optionally, this can be used for dynamic imports of the correct module.
