
import streamlit as st
import pandas as pd
import os

from utils import evaluate_submission
from notebook_insights import show_notebook_insights
from dse_tracker_panel import show_dse_maturity_panel
from feature_importance_compare import show_feature_importance_panel

st.set_page_config(page_title="Titanic AutoML Tracker", layout="wide")

tab1, tab2, tab3, tab6 = st.tabs([
    "📊 Leaderboard",
    "🧠 Notebook Insights",
    "📋 DSE Tracker"
    "📊 Feature Importance + Suggestions"
    "📊 Feature Importance Comparison"
])

# Tab 1: Leaderboard
with tab1:
    st.title("🚢 Titanic AutoML Tracker")
    st.markdown("Upload your submission files, compare model scores, and launch AutoML runs.")
    
    uploaded_file = st.file_uploader("Upload a Kaggle submission CSV (PassengerId, Survived)", type=["csv"])
    model_name = st.text_input("Model Name (e.g., TPOT, H2O, Orange)", "")

    if uploaded_file and model_name:
        df = pd.read_csv(uploaded_file)
        score = evaluate_submission(df)
        submission_log = {
            "Model": model_name,
            "Score": score,
            "Filename": uploaded_file.name
        }
        st.success(f"✅ {model_name} scored {score:.5f} (simulated).")
        if not os.path.exists("submissions.csv"):
            pd.DataFrame([submission_log]).to_csv("submissions.csv", index=False)
        else:
            hist = pd.read_csv("submissions.csv")
            pd.concat([hist, pd.DataFrame([submission_log])]).to_csv("submissions.csv", index=False)

    if os.path.exists("submissions.csv"):
        st.subheader("📈 Submission History")
        history = pd.read_csv("submissions.csv")
        st.dataframe(history.sort_values("Score", ascending=False).reset_index(drop=True))

# Tab 2: Notebook Insights
with tab2:
    show_notebook_insights()

# Tab 3: DSE Tracker
with tab3:
    show_dse_maturity_panel()

with tab6:
    show_feature_importance_panel()

with tab7:
    show_feature_importance_panel()

# Commented out tabs (to be re-enabled later)
# from autofe import show_autofe_playground
# from automl_launcher import show_automl_launcher
# from algo_selector_dual_panel import show_algo_selector
