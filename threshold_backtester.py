# threshold_backtester.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, accuracy_score


def run_threshold_backtester():
    st.header("🔁 Auto Backtesting for Classifier Thresholds")

    if "y_test" not in st.session_state or "y_pred_proba" not in st.session_state:
        st.warning("⚠️ No TPOT or model predictions found. Run AutoML first.")
        return

    y_true = st.session_state["y_test"]
    y_scores = st.session_state["y_pred_proba"]

    if isinstance(y_scores, pd.DataFrame):
        y_scores = y_scores.iloc[:, 1] if y_scores.shape[1] > 1 else y_scores.iloc[:, 0]

    st.subheader("📈 Threshold Performance Sweep")
    thresholds = np.linspace(0.0, 1.0, 100)
    metrics = []
    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        precision = np.round(np.nan_to_num(precision_score(y_true, preds)), 3)
        recall = np.round(np.nan_to_num(recall_score(y_true, preds)), 3)
        f1 = np.round(f1_score(y_true, preds), 3)
        acc = np.round(accuracy_score(y_true, preds), 3)
        metrics.append((t, precision, recall, f1, acc))

    metric_df = pd.DataFrame(metrics, columns=["Threshold", "Precision", "Recall", "F1", "Accuracy"])
    st.line_chart(metric_df.set_index("Threshold"))

    st.subheader("🔍 Best Threshold Insights")
    best_f1_idx = metric_df["F1"].idxmax()
    best_threshold = metric_df.loc[best_f1_idx, "Threshold"]
    st.metric("Best Threshold (F1)", f"{best_threshold:.2f}")
    st.dataframe(metric_df.sort_values("F1", ascending=False).head(10))

    st.subheader("📊 Confusion Matrix at Best Threshold")
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    import matplotlib.pyplot as plt

    preds = (y_scores >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    st.pyplot(fig)

    st.caption("Simulated sweep of threshold impact on classification performance. Add cross-validation in future version.")
