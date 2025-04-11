import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import LabelEncoder

def run_dse_maturity_panel(df=None, model=None):
    st.title("🎯 Full DSE Maturity Tracker")

    # Define DSE Grid
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

    # Setup DSE matrix status tracking in session state
    if "dse_matrix_status" not in st.session_state:
        st.session_state.dse_matrix_status = {
            cat: {desc: "❌" for desc in tasks} for cat, tasks in dse_grid
        }

    color_map = {"❌": "gray", "🟡": "orange", "✅": "green"}
    cycle = ["❌", "🟡", "✅"]
    completed = []

    # Iterate through the DSE grid to display the tasks and buttons
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
            if current == "✅":
                completed.append(desc)
        st.markdown("---")

    total = sum(len(items) for _, items in dse_grid)
    pct = int(len(completed) / total * 100)
    st.markdown(f"## 🎯 DSE Maturity Completion: `{pct}%`")
    st.progress(pct / 100)

    # === AI Assistant Suggestions ===
    st.markdown(""" ### 🤖 Smart AI Suggestions """)
    if pct < 40:
        st.info("📌 Focus on completing core features and setting up reliable CI workflows first.")
    elif pct < 70:
        st.info("📌 You’re halfway there — now polish documentation, licensing, and tab structure.")
    else:
        st.success("🎉 You’re close to acquisition-ready. Consider branding, pitch decks, and strategic outreach.")

    # === How AI Insights Benefit DSE ===
    st.markdown(""" ### 🧠 AI Insights in DSE """)
    st.write("""
    **AI insights** play a crucial role in optimizing the DSE process:
    
    - **Feature Selection**: Use AI to automatically rank features based on importance and predictive power. AI can help streamline the feature engineering process by identifying the most relevant features.
    - **Hyperparameter Tuning (HPO)**: AI-driven HPO can fine-tune models to achieve optimal performance by searching through large parameter spaces more efficiently than manual methods.
    - **Data Quality**: AI can suggest preprocessing steps such as normalizing or transforming data, detecting outliers, and identifying features that may require special handling.
    - **Model Evaluation**: AI can analyze the performance of models across different subsets of data and recommend which models are best suited for particular types of problems.
    - **Validation & Drift Detection**: AI can be used to monitor model performance and automatically detect any drift in the data distribution or model predictions over time, ensuring the model remains robust after deployment.
    """)

    st.markdown("### 🧠 AI Insights: How to Leverage These Suggestions")
    st.info("""
    AI insights, when incorporated into the DSE process, guide users toward more informed decisions, faster model optimization, and robust model deployment strategies.
    
    - **Improvement Suggestions**: After completing certain tasks, AI could suggest enhancements or highlight areas that need further attention (e.g., "You're making progress in algorithm selection but could improve the 'Explainability' step").
    - **Real-Time Feedback**: As you progress through tasks in the DSE process, AI can provide feedback such as "Consider adding more training data for better feature selection" or "You may want to focus on hyperparameter tuning next based on your current progress."
    """)
