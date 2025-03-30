import streamlit as st
import pandas as pd
import os

from utils import evaluate_submission
from notebook_insights import show_notebook_insights
from feature_engineering import show_feature_engineering_playground

st.set_page_config(page_title="Titanic AutoML Tracker", layout="wide")

tab1, tab2, tab3 = st.tabs(["📊 Leaderboard", "🧠 Notebook Insights", "🧪 Feature Playground"])

with tab1:
    st.title("🚢 Titanic AutoML Tracker")
    st.markdown("Upload your submission files, compare model scores, and launch AutoML runs.")
    # (Leaderboard code goes here...)

with tab2:
    show_notebook_insights()

with tab3:
    show_feature_engineering_playground()


