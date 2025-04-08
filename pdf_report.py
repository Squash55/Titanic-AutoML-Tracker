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


def run_pdf_report():
    st.subheader("📄 Downloadable PDF Report")

    model = _tpot_cache.get("latest_tpot_model")
    X_test = _tpot_cache.get("latest_X_test")
    y_test = _tpot_cache.get("latest_y_test")

    pdf = PDFReport()
    pdf.add_page()

    if model and X_test is not None and y_test is not None:
        if is_autogluon_model(model):
            preds = model.predict(X_test)
            acc = (preds == y_test).mean()
        else:
            acc = model.score(X_test, y_test)
        pipeline_str = str(model)
        pdf.add_section("Model Accuracy", f"Accuracy on test set: {acc:.3f}")
        pdf.add_section("Model Description", pipeline_str)
    else:
        pdf.add_section("Model Info", "Model not found or not yet trained.")

    questions = get_golden_questions()
    answers = get_shap_smart_answers()
    if questions:
        qa_text = "\n\n".join([f"Q: {q}\nA: {answers.get(q, '...')}" for q in questions])
        pdf.add_section("Golden Q&A", qa_text)
    else:
        pdf.add_section("Golden Q&A", "No questions generated yet.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf.output(tmpfile.name)
        with open(tmpfile.name, "rb") as f:
            st.download_button(
                label="📥 Download Full Report",
                data=f,
                file_name="automl_report.pdf",
                mime="application/pdf"
            )
    os.remove(tmpfile.name)
