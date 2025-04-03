
# notebook_scout.py

import streamlit as st
import pandas as pd

# Simulated notebook metadata (stand-in for real Kaggle API scraping)
def load_sample_notebook_data():
    data = [
        {"Notebook": "Top Titanic Model 1", "Model": "XGBoost", "Score": 0.803, "Feature Tricks": "Title extraction, FamilySize"},
        {"Notebook": "Feature Eng. Master", "Model": "Random Forest", "Score": 0.799, "Feature Tricks": "Age imputation, Pclass/Embarked"},
        {"Notebook": "Stacking Hero", "Model": "Stacked Ensemble", "Score": 0.806, "Feature Tricks": "One-hot encoding, Feature scaling"},
        {"Notebook": "Simple Baseline", "Model": "Logistic Regression", "Score": 0.765, "Feature Tricks": "Fare bins, Sex encoding"},
        {"Notebook": "Deep Dive Titanic", "Model": "MLP", "Score": 0.775, "Feature Tricks": "Embarked fill, SibSp+Parch"},
    ]
    return pd.DataFrame(data)

def run_notebook_scout():
    st.subheader("📚 Notebook Intelligence Panel (Notebook Scout)")

    df = load_sample_notebook_data()
    st.markdown("This panel analyzes top Kaggle notebooks for strategy inspiration.")
    st.dataframe(df)

    with st.expander("📊 Insights Summary"):
        most_used_model = df['Model'].mode()[0]
        avg_score = df['Score'].mean()

        st.markdown(f"**Most-used model:** `{most_used_model}`")
        st.markdown(f"**Average leaderboard score:** `{avg_score:.3f}`")

    with st.expander("📈 Model Popularity Chart"):
        model_counts = df['Model'].value_counts()
        st.bar_chart(model_counts)

    with st.expander("🧠 Feature Engineering Highlights"):
        tricks = df['Feature Tricks'].str.split(', ').explode()
        top_tricks = tricks.value_counts().head(10)
        st.bar_chart(top_tricks)

    st.success("✅ Notebook Scout loaded successfully.")
