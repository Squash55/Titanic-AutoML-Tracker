
import streamlit as st

def show_dse_maturity_panel():
    st.title("🎯 Full DSE Maturity Tracker")

    dse_grid = [
        ("1. Create a Project Charter and Collect the right Data", [
            "Create a data dictionary, problem statement and scope of project",
            "Ensure SME & stakeholder inputs",
            "Review regulatory and ethical aspects",
            "Compare your charter against similar projects",
            "Identify the value of the project and where it will be deployed",
            "Have AI critique the completeness of the dataset"
        ]),
        ("2. Data Wrangling and Planning Data Science Experiments (DSE)", [
            "Clean data, check for errors, missing values, extreme outliers, high VIFs, & duplicates",
            "Convert categorical data to numerical when possible",
            "Create new features and establish their ranks",
            "Consider data reduction with PCA and other clustering methods",
            "Split data into train, validate, and test sets",
            "Test the value of power transformations, normalizing, standardizing / scaling data with DSEs"
        ]),
        ("3. Exploratory Data Analysis", [
            "Conduct statistical summaries & hypothesis tests between groups",
            "Create box plots, raincloud plots, scatter plots, histograms, parallel coordinate plots, time series plots, pareto charts, correlation heat maps, etc.",
            "Stratify data and compare such groups with the total data",
            "Determine the strength and direction of feature relationships",
            "Use chi-square tests for categorical variables"
        ]),
        ("4. Build, Optimize, and Verify Models", [
            "Do DSEs for feature selection, outlier and missing data handling, algo options, HPO, feature engineering, n-1 dummy encoding, one hot encoding, data transformations and normalizations, data split, CV-levels/LOOCV, expanded algorithm options, HPO, threshold tuning, ensemble / stacking methods, etc.",
            "Test and address model performance and over/over-fitting issues for train, test, & validation sets with metrics, calibration plots, residual plots, confusion matrices, AUCROC, ACPRC, etc.",
            "Compare your best model against auto-ML, AI results, & Kaggle results for similar projects",
            "Document all modeling steps and decision processes for model selection"
        ]),
        ("5. Explain Models to Stakeholders", [
            "Storytelling to include simplified model descriptions, stakeholder requirements, model performance metrics, feature rankings, SHAP, use case scenarios, comparisons to previous systems, visualizations, and descriptions of model limitations and ethical considerations",
            "Create and execute a communication plan for SMEs and stakeholders",
            "Ask AI for its best ways to explain this model"
        ]),
        ("6. Deploy, Monitor, and Retrain the Model when necessary", [
            "Monitor any data drift & concept drift over time. Set triggers for reevaluation",
            "Ask AI for best practices to identify data drift and concept drift",
            "Conduct root cause analysis on any observed model drift before retraining the model",
            "Have a plan to deal with any post-deployment issues or disaster recovery issues"
        ])
    ]

    # Setup
    if "dse_matrix_status" not in st.session_state:
        st.session_state.dse_matrix_status = {
            cat: {desc: "❌" for desc in tasks} for cat, tasks in dse_grid
        }

    color_map = {"❌": "gray", "🟡": "orange", "✅": "green"}
    cycle = ["❌", "🟡", "✅"]

    for cat, tasks in dse_grid:
        st.markdown(f"### {cat}")
        for desc in tasks:
            col = st.columns([0.05, 0.95])
            current = st.session_state.dse_matrix_status[cat][desc]
            color = color_map[current]
            if col[0].button(current, key=f"{cat}_{desc}"):
                i = cycle.index(current)
                st.session_state.dse_matrix_status[cat][desc] = cycle[(i + 1) % len(cycle)]
            col[1].markdown(f"<div style='padding:4px 0;'>{desc}</div>", unsafe_allow_html=True)
        st.markdown("---")
