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


def is_autogluon_model(model):
    return hasattr(model, "leaderboard") and hasattr(model, "predict")


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


# === PDF GENERATION LOGIC ===
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
    pdf.add_section("Model Summary", f"Model type: TPOT\nNumber of features: {X_test.shape[1]}\nTest sample size: {X_test.shape[0]}")

    # Section 2: Golden Questions & Answers
    questions = get_golden_questions(X_test)
    answers = get_shap_smart_answers(model, X_test)

    for q, a in zip(questions, answers):
        pdf.add_section(q, a)

    # Optional: Include EBM plot if selected
    if st.session_state.get("include_ebm_pdf"):
        ebm_plot_path = "ebm_feature_plot.png"
        if os.path.exists(ebm_plot_path):
            pdf.add_section("Explainable Boosting Insights", "Top features and global explanations below:")
            pdf.image(ebm_plot_path, w=180)

    # Save PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        st.success("✅ PDF Report Ready!")
        with open(tmp_file.name, "rb") as f:
            st.download_button(label="📥 Download Report", data=f, file_name="automl_summary_report.pdf")


# === ROUTING LOGIC (for app.py) ===
# Add this to app.py:
# elif subtab == "PDF Report":
#     run_pdf_report()
