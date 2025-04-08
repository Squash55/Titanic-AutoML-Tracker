# pdf_report.py

import streamlit as st
from fpdf import FPDF
import tempfile
import os
import shap
import matplotlib.pyplot as plt
import joblib

try:
    from tpot_connector import (
        latest_tpot_model,
        latest_X_train,
        latest_y_train,
        latest_X_test,
        latest_y_test
    )
except ImportError:
    latest_tpot_model = None
    latest_X_train = None
    latest_y_train = None
    latest_X_test = None
    latest_y_test = None

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


def add_shap_summary_plot(pdf, model, X_train):
    try:
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer(X_train)

        fig = plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp_img.name, bbox_inches="tight")
        plt.close(fig)
        pdf.image(tmp_img.name, w=180)
        os.remove(tmp_img.name)
    except Exception as e:
        pdf.add_section("SHAP Summary Plot", f"Error generating SHAP plot: {e}")


def generate_pdf_report():
    pdf = PDFReport()
    pdf.add_page()

    # TPOT performance summary
    if latest_tpot_model and latest_X_test is not None and latest_y_test is not None:
        acc = latest_tpot_model.score(latest_X_test, latest_y_test)
        pipeline_code = str(latest_tpot_model)

        # Executive Summary
        questions = get_golden_questions()
        answers = get_shap_smart_answers()
        sample_answers = list(answers.values())
        top_factors = ", ".join(sample_answers[:2]) if sample_answers else "[Top SHAP factors not available]"
        summary = (
            f"The model achieved an accuracy of {acc:.3f}. "
            f"Top factors influencing predictions include {top_factors}. "
            "Smart Q&A has been included to assist with diagnostics."
        )
        pdf.add_section("Executive Summary", summary)

        pdf.add_section("TPOT Model Accuracy", f"Accuracy on test set: {acc:.3f}")
        pdf.add_section("Best Pipeline Structure", pipeline_code)

        # Optional: model parameters
        try:
            params_text = str(latest_tpot_model.get_params())
            pdf.add_section("TPOT Model Parameters", params_text)
        except:
            pass
    else:
        pdf.add_section("TPOT Model", "Model not available or not trained yet.")

    # SHAP + Q&A
    questions = get_golden_questions()
    answers = get_shap_smart_answers()
    qa_summary = "\n\n".join([f"Q: {q}\nA: {answers.get(q)}" for q in questions])
    pdf.add_section("Golden Q&A (SHAP Powered)", qa_summary)

    # Global SHAP Summary
    if latest_tpot_model and latest_X_train is not None:
        add_shap_summary_plot(pdf, latest_tpot_model, latest_X_train)

    return pdf


def run_pdf_report():
    st.subheader("📄 Downloadable PDF Report")

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

    # Save model option
    st.markdown("---")
    if st.button("💾 Save Best Model to Disk"):
        save_path = os.path.join("saved_models", "best_tpot_pipeline.pkl")
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump(latest_tpot_model, save_path)
        st.success(f"Model saved to {save_path}")

    if st.button("📂 Load Saved Model"):
        load_path = os.path.join("saved_models", "best_tpot_pipeline.pkl")
        if os.path.exists(load_path):
            loaded_model = joblib.load(load_path)
            st.session_state["loaded_model"] = loaded_model
            st.success("Model loaded successfully. It will be available for use in other panels.")
        else:
            st.error("Saved model file not found.")
