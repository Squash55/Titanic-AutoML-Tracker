# golden_qa.py
import streamlit as st

# DO NOT use st.set_page_config here — only in app.py

st.title("🔮 Golden Questions + Smart Answers")

st.markdown("Choose a diagnostic question from the list below. Toggle 'Show Smart Answer' to see an expert-level insight.")

# Sample golden questions
questions = [
    "Which features most influence survival predictions?",
    "Is there a gender gap in model accuracy?",
    "What does the model struggle with most?",
    "Are there any signs of overfitting?",
    "What improvements could boost predictive performance?"
]

selected_question = st.selectbox("Select a Golden Question:", questions)
show_answer = st.checkbox("💡 Show Smart Answer")

# Display answer if requested
if show_answer:
    answers = {
        questions[0]: "Smart Answer: Based on SHAP and model importance, 'Sex', 'Pclass', and 'Fare' are the top survival predictors.",
        questions[1]: "Smart Answer: Yes, females have significantly higher survival rates, and models often learn this bias early.",
        questions[2]: "Smart Answer: Models typically struggle with middle-age males in 3rd class with lower fares — prediction uncertainty is highest here.",
        questions[3]: "Smart Answer: If training accuracy is high but test accuracy drops, overfitting is likely. Compare those scores.",
        questions[4]: "Smart Answer: Consider feature engineering (e.g., family size), scaling Fare, and trying ensemble methods like XGBoost."
    }
    st.markdown(f"### ✅ Answer")
    st.success(answers.get(selected_question, "Answer not available yet."))
else:
    st.info("Enable 'Show Smart Answer' to view an AI-generated insight.")
