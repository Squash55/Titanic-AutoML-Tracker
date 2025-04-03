
# golden_qa.py

import shap
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model, X_train


def get_shap_summary(model, X_train):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    summary_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Mean SHAP Value': abs(shap_values.values).mean(axis=0)
    }).sort_values(by='Mean SHAP Value', ascending=False)
    return summary_df


def get_golden_questions() -> list:
    return [
        "What are the top features driving the model predictions?",
        "How can we interpret the model’s decisions?",
        "Which features have the most consistent influence?",
        "Are there any features with unexpected impact?",
    ]


def get_smart_answers() -> dict:
    X, y = load_data()
    model, X_train = train_model(X, y)
    summary_df = get_shap_summary(model, X_train)

    top_features = summary_df.head(3)['Feature'].tolist()
    return {
        "What are the top features driving the model predictions?": f"The top features are: {', '.join(top_features)}.",
        "How can we interpret the model’s decisions?": "SHAP values indicate how each feature impacts predictions at the individual level.",
        "Which features have the most consistent influence?": f"Based on SHAP mean values, features like {top_features[0]} have consistent impact.",
        "Are there any features with unexpected impact?": "SHAP summary can help uncover surprising drivers that weren’t initially expected."
    }
import streamlit as st  # Add at the top if not already

def run_golden_qa():
    st.subheader("✨ Golden Q&A Panel")

    questions = get_golden_questions()
    answers = get_smart_answers()

    for q in questions:
        st.markdown(f"**Q:** {q}")
        st.markdown(f":bulb: **A:** {answers.get(q, 'No answer available.')}")

    st.success("✅ Golden Q&A loaded successfully.")
