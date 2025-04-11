# config/tabs_config.py

TITANIC_MODULE_GROUPS = {
    "🟢 Core (Data Prep)": [
        "Notebook Scout", "Auto EDA", "Auto Feature Engineering", "Distribution Auditor",
        "Outlier Suppressor", "Cat↔Reg Switcher", "LogReg Nonlinear Tricks", "LogReg + Interaction Explorer"
    ],
    "🟡 Mid (Modeling)": [
        "AutoML Comparison", "Correlation Matrix Lab", "SHAP Panel", "Golden Q&A (SHAP)", "SHAP Comparison", "SHAP Waterfall",
        "Feature Importance Lab", "SHAP Summary Lab", "Explainability Heatmap", "Explainable Boosting Visualizer"
    ],
   
    "🟣 Advanced: Optimization": [
        "Smart HPO Recommender", "DAIVID HPO Engine", "DAIVID HPO Trainer", "Zoomed HPO Explorer", "DOE Panel", "SHAP Screening Design of Experiments (DOE)", "Ensemble Builder", "Experiment Tracker & CSV Export"
    ],
    
    "🟣 Advanced: Validation & Drift": [
        "Residual Plot", "Model Diagnostics Lab", "Threshold Optimizer",
     "AI-Generated Validation Scenarios", "AutoML & AI Validation", "Feature Drift Detector", "Target Drift Diagnostic", 
    ],
    "🟣 Advanced: Stress Testing": [
        "Sensitivity Explorer", "Synthetic Perturbation Tester"
    ],
    
    "📦 Deployment & Docs": [
        "Saved Models", "PDF Report", "DAIVID Analytics Scorecard", "User Manual"
    ],
    "📑 Roadmap": [  # Ensure this section exists
        "DAIVID Roadmap"
    ]
}

DAIVID_TABS = {
    "AutoML & AI Validation": "automl_with_validation.py",
    "AI-Generated Validation Scenarios": "ai_validation_scenarios.py",
    "Auto EDA": "auto_eda.py",
    "Auto Feature Engineering": "auto_feature_engineering.py",
    "Auto FE LogReg Lab": "auto_fe_logreg_lab.py",
    "Threshold Optimizer": "auto_threshold_optimizer.py",
    "AutoML Comparison": "automl_comparison.py",
    "Cat_Reg Switcher": "catreg_switcher.py",  # Fixed name
    "DAIVID HPO Engine": "daivid_hpo_engine.py",
    "DAIVID HPO Trainer": "daivid_hpo_trainer.py",
    "DAIVID Roadmap": "daivid_roadmap.py",
    "DAIVID Analytics Scorecard": "daivid_scorecard.py",
    "Distribution Auditor": "distribution_auditor.py",
    "DOE Panel": "doe_panel.py",
    "Ensemble Builder": "ensemble_builder.py",
    "Experiment Tracker & CSV Export": "experiment_tracker.py",
    "Explainable Boosting Visualizer": "explainable_boosting_visualizer"
    "Feature Drift Detector": "feature_drift_detector.py", 
    "Feature Engineering": "feature_engineering.py",
    "Feature Impact Tester": "feature_impact_tester.py",
    "Feature Importance Compare": "feature_importance_compare.py",
    "Generate Missing Tabs": "generate_missing_tabs.py",
    "Golden Q&A (SHAP)": "golden_qna_shap.py",
    "LogReg Nonlinear Lab": "logreg_nonlinear_lab.py",
    "LogReg Nonlinear Tricks": "logreg_nonlinear_tricks.py",
    "Model Diagnostics Lab": "model_diagnostics_lab.py",
    "Model Leaderboard Panel": "model_leaderboard_panel.py",
    "Notebook Insights": "notebook_insights.py",
    "Notebook Scout": "notebook_scout.py",
    "Outlier Suppressor": "outlier_suppressor.py",
    "PDF Report": "pdf_report.py",
    "Residual Plot Panel": "residual_plot_panel.py",
    "Sensitivity Explorer": "sensitivity_explorer.py",
    "SHAP Comparison": "shap_comparison.py",
    "SHAP Screening Design of Experiments (DOE)": "shap_screening_doe.py",
    "SHAP Interpretability": "shap_interpretability.py",
    "SHAP Permutation Delta": "shap_permutation_delta.py",
    "SHAP Waterfall": "shap_waterfall.py",
    "Smart HPO Recommender": "smart_hpo_recommender.py",
    "Smart Poly Finder": "smart_poly_finder.py",
    "Synthetic Data Toggle": "synthetic_data_toggle.py",
    "Synthetic Perturbation Tester": "synthetic_perturbation_tester.py",
    "Test Imports": "test_imports.py",
    "Threshold Backtester": "threshold_backtester.py",
    "Threshold Optimizer": "threshold_optimizer.py",
    "Titanic Sample": "titanic_sample.csv",
    "TPOT Connector": "tpot_connector.py",
    "TPOT Saver": "tpot_saver.py",
    "User Manual": "user_manual.py",
    "User Manual Generator": "user_manual_generator.py",
    "Zoom HPO Explorer": "zoom_hpo_explorer.py",
}

# Optionally, this can be used for dynamic imports of the correct module.
