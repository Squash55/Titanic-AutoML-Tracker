# explainable_boosting_visualizer.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os


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

    st.subheader("Classification Report")
    y_pred = ebm.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Optional PDF inclusion toggle
    st.session_state.include_ebm_pdf = st.checkbox("🧾 Include this chart in the PDF report")

    # Save a static version of top feature plot
    try:
        feature_importances = pd.Series(ebm.feature_importances_, index=X.columns)
        feature_importances = feature_importances.sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        feature_importances.plot(kind="barh", ax=ax)
        ax.set_title("EBM Feature Importances")
        ax.set_xlabel("Importance")

        fig.tight_layout()
        fig.savefig("ebm_feature_plot.png")
        st.pyplot(fig)
        st.info("📸 Top features saved as image for PDF export.")
    except Exception as e:
        st.warning(f"⚠️ Could not save EBM plot: {e}")
