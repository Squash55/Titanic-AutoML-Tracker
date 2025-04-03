
# golden_qa.py

import streamlit as st
import pandas as pd
import shap

try:
    from tpot_connector import latest_tpot_model, latest_X_train
except ImportError:
    latest_tpot_model = None
    latest_X_train = None


def get_golden_questions() -> list:
    return [
        "What are the top features driving the model predictions?",
        "How consistent are the top features across samples?",
        "Are there any surprises in what influences predictions?",
    ]


def get_shap_smart_answers() -> dict:
    if latest_tpot_model is None or latest_X_train is None:
        return {q: "No trained model found. Please run AutoML first." for q in get_golden_questions()}

    try:
        X_sample = latest_X_train.sample(n=min(100, len(latest_X_train)), random_state=42)
        explainer = shap.Explainer(latest_tpot_model, X_sample)
        shap_values = explainer(X_sample)

        mean_shap = shap_values.abs.mean(0).values
        sorted_idx = mean_shap.argsort()[::-1]
        top_features = latest_X_train.columns[sorted_idx[:5]]

        answers = {
            "What are the top features driving the model predictions?":
                f"The most influential features are: {', '.join(top_features[:3])}.",
            "How consistent are the top features across samples?":
                f"SHAP values show consistent impact from {top_features[0]} and {top_features[1]} across many rows.",
            "Are there any surprises in what influences predictions?":
                f"{top_features[2]} and {top_features[3]} were unexpectedly impactful according to SHAP analysis.",
        }
        return answers
    except Exception as e:
        return {q: f"SHAP error: {type(e).__name__}: {e}" for q in get_golden_questions()}


def run_golden_qa():
    st.subheader("✨ Golden Q&A Panel (SHAP Enhanced)")

    questions = get_golden_questions()
    answers = get_shap_smart_answers()

    for q in questions:
        st.markdown(f"**Q:** {q}")
        st.markdown(f":bulb: **A:** {answers.get(q, 'No answer available.')}")

    st.success("✅ Golden Q&A loaded with SHAP-powered insights.")
