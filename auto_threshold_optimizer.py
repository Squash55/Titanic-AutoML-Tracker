import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tpot_connector import __dict__ as _tpot_cache
import matplotlib.pyplot as plt

def run():
    st.subheader("🎯 Auto Threshold Optimization")

    model = _tpot_cache.get("latest_tpot_model") or _tpot_cache.get("latest_rf_model")
    X_test = _tpot_cache.get("latest_X_test")
    y_test = _tpot_cache.get("latest_y_test")

    if model is None or X_test is None or y_test is None:
        st.warning("⚠️ No model or test data found. Run TPOT or RandomForest first.")
        return

    try:
        probs = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        st.error(f"❌ Could not compute probabilities: {e}")
        return

    thresholds = np.linspace(0.0, 1.0, 101)
    metrics = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        metrics.append({
            "Threshold": t,
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1 Score": f1_score(y_test, preds, zero_division=0),
            "Accuracy": accuracy_score(y_test, preds),
        })

    df = pd.DataFrame(metrics)

    metric_to_optimize = st.selectbox("Optimize for:", ["F1 Score", "Precision", "Recall", "Accuracy"])
    best_row = df.loc[df[metric_to_optimize].idxmax()]
    st.success(f"Best {metric_to_optimize}: {best_row[metric_to_optimize]:.3f} at threshold = {best_row['Threshold']:.2f}")

    fig, ax = plt.subplots()
    for m in ["F1 Score", "Precision", "Recall", "Accuracy"]:
        ax.plot(df["Threshold"], df[m], label=m)
    ax.axvline(best_row["Threshold"], linestyle="--", color="gray", label="Best Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    st.pyplot(fig)

    # Interactive slider to explore
    slider_val = st.slider("Manual Threshold", 0.0, 1.0, float(best_row["Threshold"]), 0.01)
    pred_slider = (probs >= slider_val).astype(int)

    st.markdown("### 🧪 Classification Report at Selected Threshold")
    st.write({
        "Precision": precision_score(y_test, pred_slider, zero_division=0),
        "Recall": recall_score(y_test, pred_slider, zero_division=0),
        "F1 Score": f1_score(y_test, pred_slider, zero_division=0),
        "Accuracy": accuracy_score(y_test, pred_slider)
    })

    _tpot_cache["selected_threshold"] = slider_val
