# explainable_boosting_visualizer.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap


def run_explainable_boosting_visualizer():
    st.header("📈 Explainable Boosting Visualizer")

    if "X" not in st.session_state or "y" not in st.session_state:
        st.warning("❌ No dataset found in session. Please upload or generate synthetic data.")
        return

    X = st.session_state.X
    y = st.session_state.y

    # Split for validation and interpretation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.markdown("""
    The Explainable Boosting Machine (EBM) is a glassbox model that combines accuracy with interpretability. 
    It uses generalized additive models and is particularly good for identifying non-linearities and feature interactions.
    """)

    # Train EBM
    ebm = ExplainableBoostingClassifier(random_state=42)
    ebm.fit(X_train, y_train)
    st.success("✅ EBM model trained!")

    # Show test performance
    y_pred = ebm.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)
    st.text("Classification Report:\n" + report)

    # Show global explanation plot inline
    st.subheader("🌍 Global Feature Importance")
    ebm_global = ebm.explain_global(name='EBM')
    fig = plt.figure(figsize=(10, 5))
    show(ebm_global)

    # Save image for PDF report (optional toggle)
    st.session_state.include_ebm_pdf = st.checkbox("Include this chart in the final PDF report", value=True)
    ebm_fig = ebm_global.visualize(0)
    fig_path = "ebm_feature_plot.png"
    ebm_fig.write_image(fig_path, format="png")
    st.image(fig_path, caption="Top Features from EBM", use_column_width=True)

    # Optional SHAP overlay
    st.subheader("🔍 SHAP Overlay (Optional)")
    try:
        explainer = shap.Explainer(ebm.predict, X_test)
        shap_values = explainer(X_test)
        st.set_option("deprecation.showPyplotGlobalUse", False)
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.warning(f"SHAP visualization not available: {e}")

    # Additional interpretation toggle
    st.info("EBM models provide insights comparable to decision trees while remaining highly interpretable.")
