import streamlit as st

def run_daivid_roadmap():
    st.title("🧭 DAIVID Methodology Map")
    st.markdown("""
    Welcome to the **DAIVID Copilot Framework** — a complete guided journey for AI-powered predictive analytics.

    Each phase of **DAIVID** represents a key pillar of the analytic process:
    
    - **D**: Data Exploration  
    - **A**: Algorithm Discovery  
    - **I**: Interpretability  
    - **V**: Validation & Tuning  
    - **I**: Insights & Reports  
    - **D**: Decision Simulation

    Expand each phase below to see the tools available:
    """)

    with st.expander("🔍 D: Data Exploration"):
        st.markdown("""
        - **Notebook Scout** – Browse and learn from existing Kaggle notebooks
        - **Auto EDA** – Automatic exploratory analysis
        - **Auto Feature Engineering** – AI-driven feature suggestions
        - **LogReg + Interaction Explorer** – Explore linearity and interactions
        - **Distribution Auditor** – Spot target drift and distribution shifts

        **AI Insights**: This phase emphasizes understanding the dataset. The tools here help to uncover patterns in the data and suggest which features are most impactful for modeling. The Auto Feature Engineering tool, in particular, uses AI to suggest new features that improve model performance.
        """)

    with st.expander("🧠 A: Algorithm Discovery"):
        st.markdown("""
        - **AutoML Launcher** – Run TPOT, H2O, and other frameworks
        - **Algorithm Selector** – Pros/cons, guidance for each model

        **AI Insights**: This phase is all about discovering the best algorithms for your data. AutoML frameworks, such as TPOT and H2O, provide automated search for the most suitable models, while the Algorithm Selector offers deep insights into model strengths and weaknesses. Understanding these helps in selecting the right approach for further optimization.
        """)

    with st.expander("🔬 I: Interpretability"):
        st.markdown("""
        - **SHAP Panel** – SHAP summary plots and force plots
        - **SHAP Comparison** – Compare SHAP across models
        - **SHAP Waterfall** – Instance-level SHAP explanations

        **AI Insights**: Interpretability is crucial for understanding model behavior. SHAP (SHapley Additive exPlanations) provides insights into how each feature impacts predictions. By comparing SHAP values across models, we can identify which models are most explainable, aiding in transparency and trust in AI solutions.
        """)

    with st.expander("⚙️ V: Validation & Tuning"):
        st.markdown("""
        - **DAIVID HPO Engine** – Smart HPO configuration
        - **DAIVID HPO Trainer** – Run optimization
        - **Zoomed HPO Explorer** – Recursive refinement
        - **Threshold Optimizer** – Tune classification thresholds
        - **AutoML Comparison** – Compare performance across models
        - **Ensemble Builder** – Try stacking and voting

        **AI Insights**: In this phase, we focus on improving model performance. Hyperparameter optimization (HPO) helps to find the best configuration, while tuning models and combining them into ensembles can dramatically improve predictive power. These tools ensure that the final model is both robust and accurate.
        """)

    with st.expander("📊 I: Insights & Reports"):
        st.markdown("""
        - **Golden Q&A** – Smart questions + SHAP answers
        - **PDF Report** – Exportable report
        - **Experiment Tracker** – Track runs and versions

        **AI Insights**: The insights generated in this phase provide actionable intelligence. The Golden Q&A tool uses SHAP explanations to ask smart questions about your data and model performance, while the Experiment Tracker ensures that all iterations are documented for continuous improvement.
        """)

    with st.expander("🧪 D: Decision Simulation"):
        st.markdown("""
        - **DOE Panel** – Design of Experiments simulator
        - **Saved Models** – Load/export models

        **AI Insights**: In this final phase, we focus on simulating real-world decisions based on model outcomes. The Design of Experiments (DOE) Panel helps explore how changes in input variables affect model predictions. It allows users to simulate various scenarios and see potential impacts, which aids in decision-making.
        """)

    st.success("Use the sidebar to jump into any DAIVID phase module.")
