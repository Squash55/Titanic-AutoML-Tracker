# threshold_optimizer.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def run_threshold_optimizer(y_true=None, y_proba=None):
    st.markdown("""
    ## 🎯 Threshold Optimizer
    Tune the classification threshold to optimize your model's performance based on your goals.
    """)

    if y_true is None or y_proba is None:
        st.warning("Please pass both true labels and predicted probabilities to this panel.")
        return

    thresholds = np.linspace(0, 1, 101)
    precision, recall, f1, accuracy = [], [], [], []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precision.append(precision_score(y_true, y_pred, zero_division=0))
        recall.append(recall_score(y_true, y_pred, zero_division=0))
        f1.append(f1_score(y_true, y_pred, zero_division=0))
        accuracy.append(accuracy_score(y_true, y_pred))

    best_f1_index = int(np.argmax(f1))
    best_threshold = thresholds[best_f1_index]

    st.markdown(f"**Optimal Threshold for F1 Score:** `{best_threshold:.2f}`")
    st.slider("🔧 Select Custom Threshold", 0.0, 1.0, best_threshold, step=0.01, key="custom_thresh")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, precision, label='Precision')
    ax.plot(thresholds, recall, label='Recall')
    ax.plot(thresholds, f1, label='F1 Score')
    ax.plot(thresholds, accuracy, label='Accuracy')
    ax.axvline(best_threshold, color='gray', linestyle='--', label='Best F1 Threshold')
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metric Scores vs Classification Threshold")
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    ### 🧠 Recommendation
    Choose a threshold based on the metric most aligned with your mission (e.g., maximize recall for safety-critical, or F1 for balanced tradeoff).
    """)
