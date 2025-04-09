# model_diagnostics_lab.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tpot_connector import _tpot_cache
st.write("🛠️ Loaded Model Diagnostics Lab module.")

def run_model_diagnostics_lab():
    st.title("🔬 Model Diagnostics Lab")

    model = _tpot_cache.get("latest_tpot_model")
    X_test = _tpot_cache.get("latest_X_test")
    y_test = _tpot_cache.get("latest_y_test")

    if model is None or X_test is None or y_test is None:
        st.warning("⚠️ Missing model or test data. Please run AutoML first.")
        return

    try:
        y_pred = model.predict(X_test)
        st.success("✅ Predictions generated successfully.")

        # 📋 Classification report
        st.markdown("### 📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))

        # 📉 Confusion Matrix
        st.markdown("### 📉 Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        st.pyplot(fig)

        # 📈 Prediction Breakdown
        st.markdown("### 📈 Prediction Breakdown")
        pred_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred
        })
        st.dataframe(pred_df.head(20))

        # 🔍 Class distribution
        st.markdown("### 🔍 Class Distribution (Actual vs Predicted)")
        fig2, ax2 = plt.subplots()
        sns.countplot(x="variable", hue="value", 
                      data=pd.melt(pred_df.apply(pd.Series.value_counts).fillna(0).T.reset_index(),
                                   id_vars="index"), ax=ax2)
        ax2.set_title("Distribution of Predicted vs Actual")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Error during diagnostics: {type(e).__name__}: {e}")
