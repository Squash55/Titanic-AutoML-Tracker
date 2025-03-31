import streamlit as st
import pandas as pd
import os

from utils import evaluate_submission
from notebook_insights import show_notebook_insights
# from autofe import show_autofe_playground
from feature_engineering import show_feature_engineering_playground
# from automl_launcher import show_automl_launcher
from dse_tracker_panel import show_dse_maturity_panel
# from algo_selector_dual_panel import show_algo_selector

st.set_page_config(page_title="Titanic AutoML Tracker", layout="wide")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Leaderboard",
    "🧠 Notebook Insights",
    "🧪 Feature Playground",
    "🚀 AutoML Launcher",
    "📋 DSE Tracker",
    "🧬 Algorithms + HPO"
])


with tab1:
    st.title("🚢 Titanic AutoML Tracker")
    st.markdown("Upload your submission files, compare model scores, and launch AutoML runs.")
    # (Leaderboard code goes here...)

with tab2:
    show_notebook_insights()

# with tab3:
    show_autofe_playground()

# with tab4:
    show_automl_launcher()

with tab5:
    show_dse_maturity_panel()

# with tab6:
    show_algo_selector()
