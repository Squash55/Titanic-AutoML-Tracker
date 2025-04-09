# pdf_report.py
import streamlit as st
from fpdf import FPDF
import tempfile
import os
import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd

try:
    from tpot_connector import (
        latest_tpot_model,
        latest_X_train,
        latest_y_train,
        latest_X_test,
        latest_y_test,
        _tpot_cache
    )
except ImportError:
    latest_tpot_model = None
    latest_X_train = None
    latest_y_train = None
    latest_X_test = None
    latest_y_test = None
    _tpot_cache = {}

from golden_qa import get_golden_questions, get_shap_smart_answers
from model_leaderboard_panel import run_model_leaderboard_panel


class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "AutoML + SHAP + Q&A Summary Report", ln=True, align="C")

    def add_section(self, title, content):
        self.ln(10)
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 8, content)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")


def run_pdf_report():
    st.header("🧾 Generate PDF Report")

    if latest_tpot_model is None or latest_X_test is None:
        st.warning("⚠️ No TPOT model or test data found.")
        return

    model = latest_tpot_model
    X_test = latest_X_test
    y_test = latest_y_test

    pdf = PDFReport()
    pdf.add_page()

    # Section 1: Model Overview
    pdf.add_section(
        "Model Summary",
        f"Model type: TPOT\nNumber of features: {X_test.shape[1]}\nTest sample size: {X_test.shape[0]}"
    )

    # Section 2: Golden Questions & Answers
    questions = get_golden_questions(X_test)
    answers = get_shap_smart_answers(model, X_test)

    for q, a in zip(questions, answers):
        pdf.add_section(q, a)

    # Optional: Include Explainable Boosting Plot
    if st.session_state.get("include_ebm_pdf"):
        ebm_plot_path = "ebm_feature_plot.png"
        if os.path.exists(ebm_plot_path):
            pdf.add_section(
                "Explainable Boosting Insights",
                "These plots reflect global behavior of top features in an Explainable Boosting Machine.\n"
                "Each curve represents the marginal effect of a feature on the prediction outcome, helping you understand the direction and shape of relationships."
            )
            pdf.image(ebm_plot_path, w=180)

    # Optional: Include SHAP vs Permutation Delta Plot
    if st.session_state.get("include_shap_perm_delta_pdf"):
        plot_path = st.session_state.get("shap_perm_delta_plot_path")
        if plot_path and os.path.exists(plot_path):
            pdf.add_section(
                "SHAP vs Permutation Importance Delta",
                "This chart highlights differences between SHAP and permutation feature importance values.\n"
                "Large deltas may indicate non-linear interactions, multicollinearity, or SHAP revealing effects missed by permutation testing.\n"
                "Use this to guide feature investigation and model refinement."
            )
            pdf.image(plot_path, w=180)
    # === Sensitivity Explorer Section ===
    if st.session_state.get("include_sensitivity_pdf", False):
        try:
            from sensitivity_explorer import run_sensitivity_explorer
            st.markdown("## 📐 Sensitivity Explorer")
            st.markdown("This section captures a snapshot of your custom what-if input and the resulting prediction.")
    
            model = _tpot_cache.get("latest_tpot_model")
            X_train = _tpot_cache.get("latest_X_train")
    
            if model is not None and X_train is not None:
                st.write("⬇️ Sample prediction from last configured simulation (if available):")
    
                # Reconstruct the most recent input
                user_input = {}
                for col in X_train.columns:
                    user_input[col] = st.session_state.get(f"sens_input_{col}", "—")
    
                input_df = pd.DataFrame([user_input])
                st.dataframe(input_df)
    
                try:
                    pred = model.predict(input_df)[0]
                    st.write(f"**Prediction:** {pred}")
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(input_df)[0]
                        proba_df = pd.DataFrame({"Class": model.classes_, "Probability": proba})
                        st.dataframe(proba_df)
                except Exception as e:
                    st.warning(f"Prediction failed in report: {e}")
            else:
                st.info("No AutoML model or training data available for Sensitivity Explorer report.")
        except Exception as e:
            st.error(f"Sensitivity Explorer section failed: {e}")

    # Save PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        st.success("✅ PDF Report Ready!")
        with open(tmp_file.name, "rb") as f:
            st.download_button(label="📥 Download Report", data=f, file_name="automl_summary_report.pdf")
