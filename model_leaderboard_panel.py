# model_leaderboard_panel.py

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from datetime import datetime
from tpot_connector import _tpot_cache

# ✅ Ensure model_times and durations caches exist
if "model_times" not in _tpot_cache:
    _tpot_cache["model_times"] = {}
if "model_durations" not in _tpot_cache:
    _tpot_cache["model_durations"] = {}
if "model_sources" not in _tpot_cache:
    _tpot_cache["model_sources"] = {}


def run_model_leaderboard_panel():
    st.title("🏆 Model Leaderboard Tracker")

    models = _tpot_cache.get("all_models", {})
    X_test = _tpot_cache.get("X_test")
    y_test = _tpot_cache.get("y_test")
    X_train = _tpot_cache.get("X_train")

    rows = []

    for name, model in models.items():
        acc = "-"
        shap_total = "-"
        feature_count = len(X_train.columns) if X_train is not None else "-"
        dataset_size = len(X_train) if X_train is not None else "-"
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

        if name not in _tpot_cache["model_times"]:
            _tpot_cache["model_times"][name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if name not in _tpot_cache["model_durations"]:
            _tpot_cache["model_durations"][name] = "-"
        if name not in _tpot_cache["model_sources"]:
            if hasattr(model, "leaderboard"):
                _tpot_cache["model_sources"][name] = "AutoGluon"
            elif "TPOT" in str(type(model)):
                _tpot_cache["model_sources"][name] = "TPOT"
            else:
                _tpot_cache["model_sources"][name] = "Loaded"

        timestamp = _tpot_cache["model_times"].get(name, "-")
        duration = _tpot_cache["model_durations"].get(name, "-")
        source = _tpot_cache["model_sources"].get(name, "-")

        rows.append({
            "Model Name": name,
            "Type": type(model).__name__,
            "Accuracy": acc,
            "SHAP Total": shap_total,
            "Feature Count": feature_count,
            "Dataset Size": dataset_size,
            "Trained At": timestamp,
            "Duration": duration,
            "Source": source
        })

    if rows:
        df = pd.DataFrame(rows)

        # ⭐ Highlight best model row
        numeric_df = df[pd.to_numeric(df["Accuracy"], errors="coerce").notna()].copy()
        if not numeric_df.empty:
            best_idx = numeric_df["Accuracy"].astype(float).idxmax()
            df.loc[best_idx, "Model Name"] += " 🥇"

        # 🔍 Add filters
        with st.expander("🔍 Filter Options", expanded=False):
            filter_source = st.multiselect("Filter by Source", options=df["Source"].unique().tolist(), default=df["Source"].unique().tolist())
            shap_threshold = st.slider("Minimum SHAP Total", min_value=0.0, max_value=float(df["SHAP Total"].max() or 100.0), value=0.0, step=1.0)

        df_filtered = df[(df["Source"].isin(filter_source)) & (pd.to_numeric(df["SHAP Total"], errors="coerce") >= shap_threshold)]
        st.dataframe(df_filtered, use_container_width=True)

        st.markdown("### 📊 Accuracy & SHAP Rankings")
        try:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            df_filtered_sorted_acc = df_filtered.sort_values("Accuracy", ascending=False)
            df_filtered_sorted_shap = df_filtered.sort_values("SHAP Total", ascending=False)
            ax[0].barh(df_filtered_sorted_acc["Model Name"][::-1], df_filtered_sorted_acc["Accuracy"][::-1])
            ax[0].set_title("Models Ranked by Accuracy")
            ax[1].barh(df_filtered_sorted_shap["Model Name"][::-1], df_filtered_sorted_shap["SHAP Total"][::-1], color="orange")
            ax[1].set_title("Models Ranked by SHAP Total")
            st.pyplot(fig)
        except:
            st.warning("Ranking plots unavailable. Check for missing values.")

        st.markdown("### 📈 SHAP vs Accuracy")
        if df_filtered["SHAP Total"].dtype != object and df_filtered["Accuracy"].dtype != object:
            fig, ax = plt.subplots()
            ax.scatter(df_filtered["SHAP Total"], df_filtered["Accuracy"], s=100)
            for i, row in df_filtered.iterrows():
                ax.text(row["SHAP Total"], row["Accuracy"], row["Model Name"], fontsize=9)
            ax.set_xlabel("Total SHAP Value")
            ax.set_ylabel("Accuracy")
            ax.set_title("Model Interpretability vs Accuracy")
            st.pyplot(fig)

        st.markdown("### ⏳ Training Timeline")
        try:
            df_filtered["Trained At Parsed"] = pd.to_datetime(df_filtered["Trained At"], errors="coerce")
            df_sorted = df_filtered.sort_values("Trained At Parsed")
            fig, ax = plt.subplots()
            ax.plot(df_sorted["Trained At Parsed"], df_sorted["Accuracy"], marker="o")
            for i, row in df_sorted.iterrows():
                ax.text(row["Trained At Parsed"], row["Accuracy"], row["Model Name"], fontsize=8)
            ax.set_xlabel("Training Time")
            ax.set_ylabel("Accuracy")
            ax.set_title("Model Accuracy Over Time")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            fig.autofmt_xdate()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Timeline plot error: {e}")

        if st.button("📥 Export Leaderboard to CSV"):
            st.download_button(
                "Download CSV",
                data=df_filtered.drop(columns=["Trained At Parsed"], errors="ignore").to_csv(index=False).encode(),
                file_name="model_leaderboard.csv",
                mime="text/csv"
            )
    else:
        st.info("No models have been trained yet. Run AutoML to populate leaderboard.")
