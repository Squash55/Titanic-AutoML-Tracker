# pdf_report.py
import streamlit as st
from fpdf import FPDF
import tempfile
import os
import shap
import matplotlib.pyplot as plt
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


def generate_pdf_report():
    pdf = PDFReport()
    pdf.add_page()

    # TPOT performance summary
    if latest_tpot_model and latest_X_test is not None and latest_y_test is not None:
        acc = latest_tpot_model.score(latest_X_test, latest_y_test)
        pipeline_code = str(latest_tpot_model)
        pdf.add_section("TPOT Model Accuracy", f"Accuracy on test set: {acc:.3f}")
        pdf.add_section("Best Pipeline Structure", pipeline_code)
    else:
        pdf.add_section("TPOT Model", "Model not available or not trained yet.")

    # SHAP + Q&A
    questions = get_golden_questions()
    answers = get_shap_smart_answers()
    qa_summary = "\n\n".join([f"Q: {q}\nA: {answers.get(q)}" for q in questions])
    pdf.add_section("Golden Q&A (SHAP Powered)", qa_summary)

    return pdf


def run_pdf_report():
    st.subheader("📄 Downloadable PDF Report")
    st.markdown("🧪 TPOT-only mode active")

    if latest_tpot_model is None:
        st.warning("⚠️ Train a model in AutoML Launcher first.")
        return

    pdf = generate_pdf_report()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf.output(tmpfile.name)
        with open(tmpfile.name, "rb") as f:
            st.download_button(
                label="📥 Download Report",
                data=f,
                file_name="AutoML_SHAP_Report.pdf",
                mime="application/pdf"
            )
    os.remove(tmpfile.name)
