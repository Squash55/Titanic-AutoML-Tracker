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

        st.markdown("### 📌 Rule-Based Summary")
        if not df_filtered.empty:
            best_accuracy = df_filtered.loc[df_filtered["Accuracy"].astype(float).idxmax()]
            st.success(f"Top Accuracy: {best_accuracy['Model Name']} ({best_accuracy['Accuracy']:.3f})")
            best_shap = df_filtered.loc[df_filtered["SHAP Total"].astype(float).idxmax()]
            st.info(f"Most Interpretable: {best_shap['Model Name']} (SHAP: {best_shap['SHAP Total']:.1f})")

        st.markdown("### 🤖 GPT Insight Summary")
        if st.button("🧠 Generate AI Summary"):
            import openai
            try:
                openai.api_key = st.secrets["OPENAI_API_KEY"]
                prompt = f"""
                Given the following table of models with accuracy and SHAP total values, summarize the key findings.

                {df_filtered[['Model Name', 'Accuracy', 'SHAP Total', 'Source']].to_string(index=False)}

                Provide insights on which model is the best overall and which are most interpretable.
                """
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )
                summary = response.choices[0].message.content
                st.markdown(f"> {summary}")
            except Exception as e:
                st.error(f"AI summary failed: {e}")

        st.markdown("### 📥 Compare with Uploaded Leaderboard")
        uploaded_file = st.file_uploader("Upload Previous Leaderboard CSV", type=["csv"])
        if uploaded_file:
            try:
                old_df = pd.read_csv(uploaded_file)
                st.dataframe(old_df, use_container_width=True)
                st.markdown("#### 🔄 Change Detection")
                diff_cols = [col for col in ["Model Name", "Accuracy"] if col in old_df.columns and col in df.columns]
                if diff_cols:
                    merged = pd.merge(old_df, df, on="Model Name", suffixes=("_Old", "_New"))
                    merged["Accuracy Change"] = pd.to_numeric(merged["Accuracy_New"], errors="coerce") - pd.to_numeric(merged["Accuracy_Old"], errors="coerce")
                    st.dataframe(merged[["Model Name", "Accuracy_Old", "Accuracy_New", "Accuracy Change"]])
            except Exception as e:
                st.error(f"Upload comparison failed: {e}")

        if st.button("📥 Export Leaderboard to CSV"):
            st.download_button(
                "Download CSV",
                data=df_filtered.drop(columns=["Trained At Parsed"], errors="ignore").to_csv(index=False).encode(),
                file_name="model_leaderboard.csv",
                mime="text/csv"
            )
    else:
        st.info("No models have been trained yet. Run AutoML to populate leaderboard.")
