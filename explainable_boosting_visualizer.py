import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

def run_explainable_boosting_visualizer():
    st.header("📈 Explainable Boosting Visualizer")

    if "X" not in st.session_state or "y" not in st.session_state:
        st.error("❌ No dataset found. Please upload or generate synthetic data first.")
        return

    X = st.session_state.X
    y = st.session_state.y

    test_size = st.slider("Test size (for validation)", 0.1, 0.5, 0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    with st.spinner("Training Explainable Boosting Classifier..."):
        ebm = ExplainableBoostingClassifier(random_state=0)
        ebm.fit(X_train, y_train)

    st.success("✅ EBM model trained!")

    st.subheader("Top Global Explanations")
    ebm_global = ebm.explain_global()
    show(ebm_global)

    # Generate model evaluation metrics
    y_pred = ebm.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Display AI insights based on model performance
    accuracy = report["accuracy"]
    st.markdown("### 🤖 AI Insights")

    if accuracy < 0.8:
        st.warning(f"⚠️ The model accuracy is **{accuracy*100:.2f}%**, which is below the ideal threshold of 80%. Consider adding more features, fine-tuning hyperparameters, or trying a different model.")
    elif accuracy < 0.9:
        st.info(f"📊 The model accuracy is **{accuracy*100:.2f}%**. It's performing decently, but you may want to consider additional optimization steps or model ensembling.")
    else:
        st.success(f"🎉 Excellent model accuracy: **{accuracy*100:.2f}%**! You can proceed with model interpretation and deployment.")

    # Dynamic Insights Based on SHAP Values
    st.subheader("🔍 SHAP Value Insights")
    explainer = shap.Explainer(ebm, X_train)
    shap_values = explainer(X_test)
    
    # Display SHAP values for a specific feature
    feature_importances = pd.Series(ebm.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=True)
    
    st.markdown("### 📊 Top Feature Importances")
    fig, ax = plt.subplots(figsize=(8, 6))
    feature_importances.plot(kind="barh", ax=ax)
    ax.set_title("EBM Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    st.pyplot(fig)
    
    # AI Insight Based on Feature Importance
    st.markdown("### 🧠 AI Insights: Feature Importance")
    if feature_importances.max() > 0.2:
        st.warning("⚠️ One feature is dominating the model's prediction. Consider investigating the impact of this feature and whether it could lead to overfitting.")
    else:
        st.success("✅ The model is utilizing a balanced set of features. Great diversity in feature importance.")

    # Show SHAP dependence plots
    st.subheader("SHAP Dependence Plots for Top Features")
    for feature in feature_importances.index[:3]:
        st.markdown(f"#### {feature} SHAP Dependence Plot")
        shap.dependence_plot(feature, shap_values.values, X_test)
        st.pyplot()

    # Optional PDF inclusion toggle
    st.session_state.include_ebm_pdf = st.checkbox("🧾 Include this chart in the PDF report")

    # Real-time Model Tuning Feedback
    st.markdown("### 💡 Model Tuning Insights")
    if "model_tuning" in st.session_state:
        tuning_feedback = st.session_state.model_tuning
        st.info(f"🔧 **Model Tuning Suggestions**: {tuning_feedback}")
    else:
        st.info("🔧 **Model Tuning Suggestions**: Consider using hyperparameter optimization to further improve model accuracy.")

    # AI Insights Based on Performance
    if accuracy < 0.75:
        st.info("🔧 Consider adding more diverse features or experimenting with different transformations (e.g., normalization, log-transforms) for better results.")
    
    # Advanced AI Insights
    st.markdown("### 🧠 Advanced AI Insights")
    st.write("""
    **AI insights** are crucial in optimizing the EBM model:
    - **Feature Selection**: Use AI to prioritize features and identify any redundant or irrelevant features. Removing unnecessary features can help reduce model complexity and improve performance.
    - **Hyperparameter Tuning**: AI can recommend hyperparameter values that maximize model performance, saving time compared to manual tuning.
    - **Model Evaluation**: Use AI-driven performance evaluation techniques like cross-validation and ensemble learning to identify potential weaknesses or overfitting in the model.
    - **Model Drift Detection**: After deployment, AI can monitor model drift and suggest re-training or feature updates to maintain optimal performance.
    """)

    st.markdown("### 🧠 AI Insights: How to Leverage These Suggestions")
    st.info("""
    AI insights, when incorporated into the model building process, guide users towards more informed decisions, faster model optimization, and robust deployment strategies.
    
    - **Improvement Suggestions**: After completing certain tasks, AI could suggest enhancements or highlight areas that need further attention (e.g., "You're making progress in algorithm selection but could improve the 'Explainability' step").
    - **Real-Time Feedback**: As you progress through tasks in the DSE process, AI can provide feedback such as "Consider adding more training data for better feature selection" or "You may want to focus on hyperparameter tuning next based on your current progress."
    """)
