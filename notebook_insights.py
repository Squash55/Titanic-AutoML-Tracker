
import streamlit as st

def show_notebook_insights():
    st.header("🧠 Notebook Intelligence Panel")
    st.markdown("Curated insights from top Kaggle Titanic submissions.")

    notebooks = [
        {
            "title": "1. Titanic Top 1% EDA + Feature Engineering + XGBoost",
            "score": "0.80383",
            "features": [
                "Title from Name",
                "FamilySize = SibSp + Parch + 1",
                "IsAlone (FamilySize == 1)",
                "FareBin (quartile binning)",
                "AgeGroup (categorized age)"
            ],
            "models": ["XGBoost", "Voting Classifier (XGB + RF + SVC)"],
            "link": "https://www.kaggle.com/code/startupsci/titanic-data-science-solutions"
        },
        {
            "title": "2. High Score with Simple Models & Clean Features",
            "score": "0.79904",
            "features": [
                "Sex encoded as binary",
                "Age imputed with median per title",
                "CabinKnown as binary feature"
            ],
            "models": ["Random Forest", "Logistic Regression"],
            "link": "https://www.kaggle.com/code/jesucristo/1st-place-solution-a-complete-guide"
        },
        {
            "title": "3. Titanic - Top 5% with AutoML + Feature Tuning",
            "score": "0.79979",
            "features": [
                "Embarked filled with mode",
                "Pclass encoded",
                "Fare categorized into 5 bins"
            ],
            "models": ["AutoML (TPOT)", "Stacked Voting Ensemble"],
            "link": "https://www.kaggle.com/code/omarelgabry/titanic"
        },
        {
            "title": "4. Feature-Rich Logistic Regression Approach",
            "score": "0.80120",
            "features": [
                "Title mapped into major groups",
                "Fare per person",
                "Deck from Cabin"
            ],
            "models": ["Logistic Regression"],
            "link": "https://www.kaggle.com/code/helgejo/an-interactive-data-science-tutorial"
        },
        {
            "title": "5. Neural Net with Feature Engineering",
            "score": "0.79692",
            "features": [
                "Polynomial Age & Fare",
                "Label encoding for all categoricals"
            ],
            "models": ["Keras Neural Net"],
            "link": "https://www.kaggle.com/code/nadintamer/titanic-survival-predictions-beginner"
        }
    ]

    for note in notebooks:
        with st.expander(f"{note['title']} (Score: {note['score']})"):
            st.markdown("**Key Features Used:**")
            for feature in note['features']:
                st.markdown(f"- {feature}")
            st.markdown("**Model(s):** " + ", ".join(note['models']))
            st.markdown(f"[🔗 View Full Notebook]({note['link']})")
