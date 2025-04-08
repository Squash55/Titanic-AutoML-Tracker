# residual_plot_panel.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from tpot_connector import _tpot_cache


def run_residual_plot_panel():
    st.title("📉 Residual Plot Visualizer")

    model = _tpot_cache.get("latest_tpot_model")
    X_test = _tpot_cache.get("latest_X_test")
    y_test = _tpot_cache.get("latest_y_test")

    if model is None or X_test is None or y_test is None:
        st.warning("⚠️ No trained model or test data found. Please run AutoML first.")
        return

    st.markdown("This panel shows residual plots to help diagnose model fit, variance, and potential outliers.")

    try:
        predictions = model.predict(X_test)
        residuals = y_test - predictions

        df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": predictions,
            "Residuals": residuals,
        })

        st.markdown("### 🔬 Residual Distribution Plot")
        fig1, ax1 = plt.subplots()
        sns.histplot(df["Residuals"], bins=30, kde=True, ax=ax1)
        ax1.axvline(0, color="red", linestyle="--")
        ax1.set_title("Distribution of Residuals")
        st.pyplot(fig1)

        st.markdown("### 📈 Residuals vs Predicted Plot")
        fig2, ax2 = plt.subplots()
        ax2.scatter(df["Predicted"], df["Residuals"], alpha=0.7)
        ax2.axhline(0, color="red", linestyle="--")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residuals vs Predicted")
        st.pyplot(fig2)

        st.markdown("### 📊 Residuals Table Sample")
        st.dataframe(df.head(10))

        st.markdown("### 📦 Export Residuals to CSV")
        st.download_button("Download CSV", df.to_csv(index=False).encode(), "residuals.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ Residual analysis failed: {e}")
