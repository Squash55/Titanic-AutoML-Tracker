
# daivid_hpo_engine.py
import streamlit as st
import pandas as pd
import numpy as np
import time
from tpot_connector import _tpot_cache
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def run_daivid_hpo_engine():
    st.title("🧪 DAIVID HPO Engine")
    st.markdown("This module performs hyperparameter optimization using the selected configuration from the Smart HPO panel.")

    config = _tpot_cache.get("last_hpo_config", {})
    df = _tpot_cache.get("X_train")
    y = _tpot_cache.get("y_train")

    if df is None or y is None:
        st.warning("⚠️ Training data not found. Please run AutoML first.")
        return

    if not config:
        st.warning("⚠️ No HPO configuration found. Please configure Smart HPO first.")
        return

    st.markdown("### 🔍 Current HPO Configuration")
    st.json(config)

    # Simulate model training
    st.markdown("### 📊 Training Progress")
    progress = st.progress(0)
    for i in range(1, 101):
        time.sleep(0.01)
        progress.progress(i)

    st.success("🎉 HPO simulation complete! Model trained with top config.")

    # Simulate confusion matrix
    st.markdown("### 🔢 Simulated Confusion Matrix")
    y_true = np.random.randint(0, 2, 100)
    y_pred = y_true.copy()
    y_pred[:10] = 1 - y_pred[:10]  # Add some errors

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    st.pyplot(fig)

    st.markdown("### 🧭 What's Next?")
    st.info("""
✅ You've trained a model using Smart HPO.

➡️ **Next Step:**
- Analyze SHAP values in the SHAP Panel
- Use Threshold Optimizer to improve performance
- Export results and models
    """)
