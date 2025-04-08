# model_leaderboard_panel.py

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import time
from datetime import datetime
from tpot_connector import _tpot_cache

# ✅ Ensure model_times cache exists
if "model_times" not in _tpot_cache:
    _tpot_cache["model_times"] = {}


def run_model_leaderboard_panel():
    st.title("🏆 Model Leaderboard Tracker")

    models = _tpot_cache.get("all_models", {})
    X_test = _tpot_cache.get("X_test")
    y_test = _tpot_cache.get("y_test")
    X_train = _tpot_cache.get("X_train")

    rows = []
    shap_scores = []

    for name, model in models.items():
        acc = "-"
        shap_total = "-"
        try:
            if hasattr(model, "predict") and X_test is not None and y_test is not None:
                if hasattr(model, "leaderboard"):
                    preds = model.predict(X_test)
                    acc = (preds == y_test).mean()
                else:
                    acc = model.score(X_test, y_test)

            if X_train is not None:
                try:
                    explainer = shap.Explainer(model.predict, X_train)
                    shap_values = explainer(X_train[:100])
                    shap_total = float(abs(shap_values.values).sum())
                except:
                    pass
        except:
            pass

        # ✅ Auto-store timestamp if missing
        if name not in _tpot_cache["model_times"]:
            _tpot_cache["model_times"][name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        timestamp = _tpot_cache["model_times"].get(name, "-")
        rows.append({
            "Model Name": name,
            "Type": type(model).__name__,
            "Accuracy": acc,
            "SHAP Total": shap_total,
            "Trained At": timestamp
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        st.markdown("### 📈 SHAP vs Accuracy")
        if df["SHAP Total"].dtype != object and df["Accuracy"].dtype != object:
            fig, ax = plt.subplots()
            ax.scatter(df["SHAP Total"], df["Accuracy"], s=100)
            for i, row in df.iterrows():
                ax.text(row["SHAP Total"], row["Accuracy"], row["Model Name"], fontsize=9)
            ax.set_xlabel("Total SHAP Value")
            ax.set_ylabel("Accuracy")
            ax.set_title("Model Interpretability vs Accuracy")
            st.pyplot(fig)

        if st.button("📥 Export Leaderboard to CSV"):
            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False).encode(),
                file_name="model_leaderboard.csv",
                mime="text/csv"
            )
    else:
        st.info("No models have been trained yet. Run AutoML to populate leaderboard.")
