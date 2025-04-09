# user_manual.py

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
from tpot_connector import _tpot_cache

def run_sensitivity_explorer():
    st.title("📐 Sensitivity Explorer (What-if Panel)")

    model = _tpot_cache.get("latest_tpot_model")
    X_train = _tpot_cache.get("latest_X_train")

    if model is None or X_train is None:
        st.warning("⚠️ No model or training data found. Please run AutoML first.")
        return

    st.markdown("""
    Adjust each feature below to simulate hypothetical inputs.
    We'll show the model's prediction and probability (if available).
    """)

    edge_case_mode = st.checkbox("🧪 Edge Case Mode", value=False)
    user_input = {}

    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            min_val = float(X_train[col].min())
            max_val = float(X_train[col].max())
            mean_val = float(X_train[col].mean())

            if edge_case_mode:
                extreme_option = st.radio(
                    f"{col} (Extreme Mode)",
                    ["Normal", "Min", "Max", "Random"],
                    horizontal=True
                )
                if extreme_option == "Min":
                    val = min_val
                elif extreme_option == "Max":
                    val = max_val
                elif extreme_option == "Random":
                    val = float(np.random.uniform(min_val, max_val))
                else:
                    val = mean_val
                user_input[col] = val
            else:
                user_input[col] = st.slider(col, min_val, max_val, mean_val)
        else:
            options = list(X_train[col].dropna().unique())
            if edge_case_mode:
                options.append("🚫 Missing")
            selected = st.selectbox(col, options)
            user_input[col] = None if selected == "🚫 Missing" else selected

    input_df = pd.DataFrame([user_input])
    st.markdown("### 🔍 Simulated Input")
    st.dataframe(input_df)

    for k, v in user_input.items():
        st.session_state[f"sens_input_{k}"] = v

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🧠 Model Prediction: **{prediction}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            st.markdown("### 📈 Prediction Probabilities")
            proba_df = pd.DataFrame({"Class": model.classes_, "Probability": proba})
            st.dataframe(proba_df)
    except Exception as e:
        st.error(f"❌ Prediction failed: {type(e).__name__}: {e}")

    st.markdown("---")
    st.markdown("""
    ### 🧠 Interpretation
    - Use this panel to test edge cases and understand prediction drivers.
    - Try adjusting only one feature at a time to isolate sensitivity.
    - Toggle **Edge Case Mode** to simulate real-world anomalies or stress tests.
    """)

    st.session_state["last_used_tab"] = "Sensitivity Explorer"
    st.session_state["include_sensitivity_pdf"] = st.sidebar.checkbox("🧪 Include Sensitivity Explorer Section", value=True)

def append_to_manual():
    st.markdown("### 📐 Sensitivity Explorer (What-if Panel)")
    st.markdown("""
    This panel allows you to simulate hypothetical scenarios by adjusting input feature values.

    - Use the **sliders and selectors** to create what-if inputs.
    - Toggle **Edge Case Mode** to test Min, Max, Random, or Missing values.
    - Get real-time predictions and probability scores based on your input.
    - Great for understanding how small changes affect outcomes.

    You can also include the current sensitivity configuration in the **PDF Report**.
    """)
    if st.session_state.get("manual_image_mode"):
        st.image("screenshots/sensitivity_explorer_demo.png", caption="Sensitivity Explorer Screenshot", use_column_width=True)

    st.markdown("### 🤖 AutoML Launcher")
    st.markdown("""
    Runs an AutoML pipeline using TPOT.

    - Automatically searches for the best preprocessing + model combo
    - Customizable generations, population size, scoring metric
    - Trains multiple models and selects the best pipeline

    Recommended as the first modeling step after EDA.
    """)
    if st.session_state.get("manual_image_mode"):
        st.image("screenshots/automl_launcher_demo.png", caption="AutoML Launcher in action", use_column_width=True)

    st.markdown("### 🔍 SHAP Comparison")
    st.markdown("""
    Compare SHAP feature importance across multiple models.

    - Helps identify consistent vs unstable feature contributions
    - Useful when comparing RandomForest, XGBoost, and others
    - Makes it easy to pick the most interpretable or robust model

    Especially useful when choosing models for deployment.
    """)
    if st.session_state.get("manual_image_mode"):
        st.image("screenshots/shap_comparison_demo.png", caption="SHAP Comparison Visual", use_column_width=True)

    st.markdown("### 🧪 DOE Panel")
    st.markdown("""
    Use Design of Experiments (DOE) to run structured input sweeps.

    - Visualize main effects and interaction effects
    - Ideal for identifying key variables and non-linear interactions
    - Based on classic factorial DOE methods

    Boosts trust and understanding of your model’s behavior under controlled tests.
    """)
    if st.session_state.get("manual_image_mode"):
        st.image("screenshots/doe_panel_demo.png", caption="DOE Visual Explorer", use_column_width=True)

    st.markdown("### 📈 Residual Plot")
    st.markdown("""
    Visual diagnostic comparing predicted vs actual outcomes.

    - Helps detect systematic bias
    - Useful for regression or probability-based classification models
    - Shows under- and over-prediction zones
    """)
    if st.session_state.get("manual_image_mode"):
        st.image("screenshots/residual_plot_demo.png", caption="Residual Plot Sample", use_column_width=True)

    st.markdown("### 🌊 Feature Drift Detector")
    st.markdown("""
    Identifies changes in feature distributions over time.

    - KS test for numeric drift, Chi² for categorical
    - Compare current data with training data
    - Highlight features that need retraining attention
    """)
    if st.session_state.get("manual_image_mode"):
        st.image("screenshots/feature_drift_demo.png", caption="Feature Drift Summary", use_column_width=True)

def run_user_manual():
    st.title("📘 DAIVID Analytics User Manual")

    with st.sidebar.expander("📘 Export Options", expanded=False):
        include_images = st.checkbox("🖼️ Include Visual Aids", value=True, key="manual_image_mode")
        export_pdf = st.button("📄 Generate PDF")
        export_md = st.button("📝 Export Markdown")

    append_to_manual()

    if export_pdf:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "DAIVID Analytics User Manual")
        pdf.ln()
        sections = [
            "Sensitivity Explorer: What-if Analysis tool with real-time prediction feedback.",
            "AutoML: Automatically builds and selects the best ML pipeline using TPOT.",
            "SHAP Comparison: Visualize feature importance variation across models.",
            "DOE Panel: Structured input testing with factor effect visualization.",
            "Residual Plot: Helps detect bias in predictions.",
            "Feature Drift Detector: Monitors distribution shifts to maintain model performance."
        ]
        for section in sections:
            pdf.multi_cell(0, 10, section)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf.output(tmp_file.name)
            with open(tmp_file.name, "rb") as f:
                st.download_button("📥 Download PDF Manual", f, file_name="DAIVID_User_Manual.pdf")

    if export_md:
        md_content = """# DAIVID Analytics User Manual

## 📐 Sensitivity Explorer
Simulate what-if scenarios using custom feature inputs.

## 🤖 AutoML Launcher
Run TPOT to find optimized pipelines.

## 🔍 SHAP Comparison
Compare model interpretability across algorithms.

## 🧪 DOE Panel
Use structured experiments to explore model behavior.

## 📈 Residual Plot
Visualize prediction errors and trends.

## 🌊 Feature Drift Detector
Monitor for changes in input distributions over time.
"""
        st.download_button("📥 Download Markdown Manual", md_content, file_name="DAIVID_User_Manual.md")
