from notebook_insights import show_notebook_insights
import streamlit as st
import pandas as pd
import os
from utils import evaluate_submission

st.set_page_config(page_title="Titanic AutoML Tracker", layout="wide")
st.title("🚢 Titanic AutoML Tracker")
st.markdown("Upload your submission files, compare model scores, and launch AutoML runs.")

# Leaderboard Upload Section
st.header("📊 Leaderboard Upload")
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

# Display Leaderboard
if os.path.exists("submissions.csv"):
    st.subheader("📈 Submission History")
    history = pd.read_csv("submissions.csv")
    st.dataframe(history.sort_values("Score", ascending=False).reset_index(drop=True))
if st.sidebar.button("🧠 Notebook Insights"):
    show_notebook_insights()
