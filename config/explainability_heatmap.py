import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def run_explainability_heatmap():
    st.header("🔥 Explainability Heatmap")

    # Stated Purpose Below the Title
    st.markdown("""
    This module helps visualize feature importance in a machine learning model using **SHAP values**.
    By displaying a heatmap of feature importance, you can interpret how each feature influences the model's predictions.
    Additionally, this tool provides real-time insights on feature performance and potential improvements for better model explainability.
    """)

    if "X" not in st.session_state or "y" not in st.session_state:
        st.error("❌ No dataset found. Please upload or generate synthetic data first.")
        return

    # Load data from session state
    X = st.session_state.X
    y = st.session_state.y

    # Split data
    test_size = st.slider("Test size (for validation)", 0.1, 0.5, 0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Select model for explainability
    model = st.selectbox("Select model for explainability", ["RandomForest", "LogisticRegression", "ExplainableBoosting"])

    if model == "RandomForest":
        model_instance = RandomForestClassifier(random_state=42)
    elif model == "LogisticRegression":
        model_instance = LogisticRegression(max_iter=1000)
    elif model == "ExplainableBoosting":
        model_instance = ExplainableBoostingClassifier(random_state=42)

    model_instance.fit(X_train, y_train)
    st.success(f"✅ {model} model trained!")

    # Use SHAP for feature importance (or model-based feature importance)
    explainer = shap.Explainer(model_instance, X_train)
    shap_values = explainer(X_test)

    # Generate a SHAP summary plot
    st.subheader("📊 SHAP Summary Plot")
    shap.summary_plot(shap_values, X_test)

    # Display Feature Importance Heatmap
    st.subheader("🔥 Feature Importance Heatmap")
    importance = shap_values.abs.mean(axis=0).values  # Absolute SHAP value to get feature importance
    importance_df = pd.DataFrame(importance, index=X.columns, columns=["Importance"])
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Create Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(importance_df.T, annot=True, cmap="coolwarm", cbar=True, xticklabels=True, yticklabels=True)
    st.pyplot(plt)

    # Display AI insights based on model performance
    st.markdown("### 🤖 AI Insights")

    # Analyze the performance and give feedback
    accuracy = classification_report(y_test, model_instance.predict(X_test), output_dict=True)["accuracy"]
    if accuracy < 0.8:
        st.warning(f"⚠️ The model accuracy is **{accuracy*100:.2f}%**, which is below the ideal threshold of 80%. Consider adding more features, fine-tuning hyperparameters, or trying a different model.")
    elif accuracy < 0.9:
        st.info(f"📊 The model accuracy is **{accuracy*100:.2f}%**. It's performing decently, but you may want to consider additional optimization steps or model ensembling.")
    else:
        st.success(f"🎉 Excellent model accuracy: **{accuracy*100:.2f}%**! You can proceed with model interpretation and deployment.")

    # Optional PDF inclusion toggle
    st.session_state.include_ebm_pdf = st.checkbox("🧾 Include this chart in the PDF report")

    # Save a static version of top feature plot
    try:
        feature_importances = pd.Series(model_instance.feature_importances_, index=X.columns)
        feature_importances = feature_importances.sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        feature_importances.plot(kind="barh", ax=ax)
        ax.set_title(f"{model} Feature Importances")
        ax.set_xlabel("Importance")

        fig.tight_layout()
        fig.savefig("feature_importance_plot.png")
        st.pyplot(fig)
        st.info("📸 Top features saved as image for PDF export.")
    except Exception as e:
        st.warning(f"⚠️ Could not save plot: {e}")

    # Real-time Model Tuning Feedback
    st.markdown("### 💡 Model Tuning Insights")
    if "model_tuning" in st.session_state:
        tuning_feedback = st.session_state.model_tuning
        st.info(f"🔧 **Model Tuning Suggestions**: {tuning_feedback}")
    else:
        st.info("🔧 **Model Tuning Suggestions**: Consider using hyperparameter optimization to further improve model accuracy.")
