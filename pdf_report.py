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


# ... [rest of the file unchanged, no need to repeat here]

# === In your sidebar routing logic ===
# Add this line to your navigation switch block:
elif subtab == "Model Leaderboard Tracker":
    run_model_leaderboard_panel()

# === And make sure to include it in your sidebar config ===
# Under this section in your app sidebar:
# "V: Validation & Variants": [ ... add this tab ]
"V: Validation & Variants": ["Threshold Optimizer", "DOE Panel", "Experiment Tracker", "Model Diagnostics Lab", "Model Leaderboard Tracker"],
