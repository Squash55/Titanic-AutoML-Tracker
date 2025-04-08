# explainable_boosting_visualizer.py
import streamlit as st
import pandas as pd
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import plotly.graph_objs as go


def run_explainable_boosting_visualizer():
    st.header("📈 Explainable Boosting Visualizer")

    st.markdown("""
    ### 🔍 What This Tool Does
    This tool trains an **Explainable Boosting Machine (EBM)** — a highly interpretable model from Microsoft's `interpret` package —
    and shows how each feature contributes to predictions.

    EBMs are powerful alternatives to black-box models like XGBoost or Random Forests, offering transparency without sacrificing much accuracy.
    """)

    if "X" not in st.session_state or "y" not in st.session_state:
        st.warning("❌ No data found. Please generate or upload data first.")
        return

    X = st.session_state.X
    y = st.session_state.y

    if len(np.unique(y)) > 10:
        st.warning("🚫 This demo currently supports classification problems only (target should have < 10 unique values). Try using the Cat ↔ Reg Switcher tab.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    with st.spinner("🔄 Training Explainable Boosting Model..."):
        ebm = ExplainableBoostingClassifier(random_state=42)
        ebm.fit(X_train, y_train)

    y_pred = ebm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Accuracy on Test Set: {acc:.3f}")

    st.markdown("### 📊 Feature Contributions (Global)")
    global_exp = ebm.explain_global()

    # Top feature bars
    top_feats = pd.DataFrame({
        "Feature": global_exp.feature_names,
        "Importance": global_exp.feature_importances_
    }).sort_values("Importance", ascending=False).head(10)

    fig = go.Figure(go.Bar(
        x=top_feats["Importance"],
        y=top_feats["Feature"],
        orientation='h',
        marker_color='indigo'
    ))
    fig.update_layout(title="Top Feature Importances (EBM)", height=400)
    st.plotly_chart(fig)

    st.markdown("### 🔍 Inspect Individual Feature Effects")
    feature = st.selectbox("Select Feature to Inspect", global_exp.feature_names)
    feature_idx = global_exp.feature_names.index(feature)
    feature_vals = global_exp.data("scores")[feature_idx]
    feature_bins = global_exp.data("names")[feature_idx]

    fig2 = go.Figure(go.Scatter(
        x=feature_bins,
        y=feature_vals,
        mode='lines+markers',
        name=feature
    ))
    fig2.update_layout(title=f"Contribution of {feature} to Prediction", xaxis_title=feature, yaxis_title="Score")
    st.plotly_chart(fig2)

    st.markdown("### 📜 Classification Report")
    st.text(classification_report(y_test, y_pred))
