# explainable_boosting_visualizer.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
import shap
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def run_explainable_boosting_visualizer():
    st.title("🧠 Explainable Boosting Visualizer")

    if "X" not in st.session_state or "y" not in st.session_state:
        st.warning("❌ No dataset found in session. Please upload or generate synthetic data first.")
        return

    X = st.session_state.X.copy()
    y = st.session_state.y.copy()

    st.info("This tool fits an Explainable Boosting Model (EBM) and displays the most important global features.")

    # Encode categorical y if classification
    if y.dtype == 'object' or y.nunique() <= 10:
        y = LabelEncoder().fit_transform(y)
        task_type = "classification"
    else:
        st.warning("EBM currently supports classification only in this version.")
        return

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Train EBM
    ebm = ExplainableBoostingClassifier(random_state=42)
    ebm.fit(X_train, y_train)

    acc = accuracy_score(y_test, ebm.predict(X_test))
    st.success(f"✅ EBM model trained. Accuracy on hold-out set: {acc:.3f}")

    # Plot global explanations
    st.subheader("🌐 Global Feature Importances")
    fig, ax = plt.subplots(figsize=(10, 5))
    ebm_global = ebm.explain_global()
    top_feats = ebm_global.data()['names'][:10]
    top_scores = ebm_global.data()['scores'][:10]
    ax.barh(top_feats[::-1], top_scores[::-1], color='skyblue')
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 10 EBM Global Features")
    st.pyplot(fig)

    # Save plot for PDF report
    plt.tight_layout()
    plt.savefig("ebm_feature_plot.png", bbox_inches="tight")

    # Optional PDF toggle
    st.session_state.include_ebm_pdf = st.checkbox("📥 Include in PDF Report", value=True)

    # Raw EBM viewer
    with st.expander("🔍 View Full EBM HTML Summary"):
        show(ebm_global)
