# pdf_report.py
import streamlit as st
from fpdf import FPDF
import tempfile
import os
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


def render_sensitivity_explorer_section(pdf):
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "📐 Sensitivity Explorer Results", ln=True)

    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, "This section simulates what-if predictions based on user-defined inputs, including edge cases.")

    input_data = {
        k.replace("sens_input_", ""): v
        for k, v in st.session_state.items()
        if k.startswith("sens_input_")
    }

    if input_data:
        pdf.ln(4)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Simulated Inputs:", ln=True)
        pdf.set_font("Arial", '', 11)
        for feature, value in input_data.items():
            pdf.cell(0, 8, f"• {feature}: {value}", ln=True)

        try:
            model = _tpot_cache.get("latest_tpot_model")
            input_df = pd.DataFrame([input_data])
            if model is not None:
                pred = model.predict(input_df)[0]
                pdf.ln(4)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"Prediction: {pred}", ln=True)

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_df)[0]
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, "Prediction Probabilities:", ln=True)
                    pdf.set_font("Arial", "", 11)
                    for cls, p in zip(model.classes_, proba):
                        pdf.cell(0, 8, f"• {cls}: {p:.3f}", ln=True)
        except Exception as e:
            pdf.cell(0, 8, f"Prediction failed: {e}", ln=True)
    else:
        pdf.cell(0, 8, "⚠️ No sensitivity inputs were configured or saved.", ln=True)


def run_pdf_report():
    st.header("🧾 Generate PDF Report")

    if latest_tpot_model is None or latest_X_test is None:
        st.warning("⚠️ No TPOT model or test data found.")
        return

    model = latest_tpot_model
    X_test = latest_X_test

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

    # Optional: Include EBM Plot
    if st.session_state.get("include_ebm_pdf"):
        ebm_plot_path = "ebm_feature_plot.png"
        if os.path.exists(ebm_plot_path):
            pdf.add_section(
                "Explainable Boosting Insights",
                "Global behavior of top EBM features. Curves show marginal effects."
            )
            pdf.image(ebm_plot_path, w=180)

    # Optional: SHAP vs Permutation Delta Plot
    if st.session_state.get("include_shap_perm_delta_pdf"):
        plot_path = st.session_state.get("shap_perm_delta_plot_path")
        if plot_path and os.path.exists(plot_path):
            pdf.add_section(
                "SHAP vs Permutation Importance Delta",
                "Highlights nonlinearities, collinearity, or signal differences. Large delta = investigate."
            )
            pdf.image(plot_path, w=180)

    # Optional: Sensitivity Explorer Section
    if st.session_state.get("include_sensitivity_pdf"):
        render_sensitivity_explorer_section(pdf)

    # Optional: User Manual Section
    if st.session_state.get("include_manual_pdf"):
        pdf.add_section("User Manual Reference", "Refer to the in-app User Manual tab for detailed guidance.")

    # Save and Offer Download
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        st.success("✅ PDF Report Ready!")
        with open(tmp_file.name, "rb") as f:
            st.download_button("📥 Download Report", data=f, file_name="automl_summary_report.pdf")
