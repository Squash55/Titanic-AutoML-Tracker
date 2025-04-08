# model_leaderboard_panel.py

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from datetime import datetime
from tpot_connector import _tpot_cache

if "model_times" not in _tpot_cache:
    _tpot_cache["model_times"] = {}
if "model_durations" not in _tpot_cache:
    _tpot_cache["model_durations"] = {}
if "model_sources" not in _tpot_cache:
    _tpot_cache["model_sources"] = {}
if "saved_models" not in _tpot_cache:
    _tpot_cache["saved_models"] = {}
if "saved_model_notes" not in _tpot_cache:
    _tpot_cache["saved_model_notes"] = {}

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
        numeric_df = df[pd.to_numeric(df["Accuracy"], errors="coerce").notna()].copy()
        if not numeric_df.empty:
            best_idx = numeric_df["Accuracy"].astype(float).idxmax()
            df.loc[best_idx, "Model Name"] += " 🥇"

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

            if st.button("📌 Promote Top Accuracy to Saved Models"):
                model_obj = models.get(best_accuracy["Model Name"].replace(" 🥇", ""))
                _tpot_cache["saved_models"][best_accuracy["Model Name"]] = model_obj
                _tpot_cache["saved_model_notes"][best_accuracy["Model Name"]] = ""
                st.success(f"✅ Saved: {best_accuracy['Model Name']}")

        st.markdown("### 💾 Saved Models Viewer")
        if _tpot_cache["saved_models"]:
            selected = st.selectbox("Select Saved Model", list(_tpot_cache["saved_models"].keys()))
            if selected:
                model = _tpot_cache["saved_models"][selected]
                st.write(f"📌 Model Name: {selected}")
                st.write(model)

                new_name = st.text_input("✏️ Rename Model", value=selected)
                new_note = st.text_area("📝 Notes for this model", value=_tpot_cache["saved_model_notes"].get(selected, ""))

                if st.button("💾 Update Name and Notes"):
                    _tpot_cache["saved_models"][new_name] = _tpot_cache["saved_models"].pop(selected)
                    _tpot_cache["saved_model_notes"][new_name] = new_note
                    if selected in _tpot_cache["saved_model_notes"]:
                        del _tpot_cache["saved_model_notes"][selected]
                    st.success("✅ Updated saved model entry")
                    st.experimental_rerun()

                if X_train is not None:
                    try:
                        explainer = shap.Explainer(model.predict, X_train)
                        shap_values = explainer(X_train[:1])
                        st.markdown("#### 🔍 SHAP Waterfall Plot (First Row)")
                        fig = shap.plots.waterfall(shap_values[0], show=False)
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"SHAP plot error: {e}")

                if st.button("📄 Generate Saved Model Report"):
                    summary = f"Model: {new_name}\n\nNotes: {new_note}\n\nType: {type(model).__name__}"
                    st.download_button("Download Report", summary.encode(), file_name=f"{new_name}_report.txt")

        else:
            st.info("No models have been promoted yet.")
        st.markdown("### 🔁 Saved vs Current Model Comparison")
        if _tpot_cache["saved_models"] and df_filtered is not None:
            compare_name = st.selectbox("Compare against this current model", df_filtered["Model Name"].tolist(), key="compare_name")
            current_row = df_filtered[df_filtered["Model Name"] == compare_name].squeeze()
            saved_row = df[df["Model Name"] == selected].squeeze()

            if not current_row.empty and not saved_row.empty:
                comp_df = pd.DataFrame({
                    "Metric": [
                        "Accuracy",
                        "SHAP Total",
                        "Feature Count",
                        "Dataset Size",
                        "Trained At",
                        "Duration",
                        "Source"
                    ],
                    "Saved Model": [
                        saved_row.get("Accuracy"),
                        saved_row.get("SHAP Total"),
                        saved_row.get("Feature Count"),
                        saved_row.get("Dataset Size"),
                        saved_row.get("Trained At"),
                        saved_row.get("Duration"),
                        saved_row.get("Source")
                    ],
                    "Current Model": [
                        current_row.get("Accuracy"),
                        current_row.get("SHAP Total"),
                        current_row.get("Feature Count"),
                        current_row.get("Dataset Size"),
                        current_row.get("Trained At"),
                        current_row.get("Duration"),
                        current_row.get("Source")
                    ]
                })
                st.table(comp_df)

        st.markdown("### 📥 Compare with Uploaded Leaderboard")
        uploaded_file = st.file_uploader("Upload Previous Leaderboard CSV", type=["csv"])
        if uploaded_file:
            try:
                old_df = pd.read_csv(uploaded_file)
                st.dataframe(old_df, use_container_width=True)
                st.markdown("#### 🔄 Change Detection with Highlighting")
                merged = pd.merge(old_df, df, on="Model Name", suffixes=("_Old", "_New"))
                merged["Accuracy Change"] = pd.to_numeric(merged["Accuracy_New"], errors="coerce") - pd.to_numeric(merged["Accuracy_Old"], errors="coerce")
                styled = merged[["Model Name", "Accuracy_Old", "Accuracy_New", "Accuracy Change"]].style.background_gradient("RdYlGn", subset=["Accuracy Change"])
                st.dataframe(styled)
            except Exception as e:
                st.error(f"Upload comparison failed: {e}")

        if st.button("📥 Export Leaderboard to CSV"):
            st.download_button(
                "Download CSV",
                data=df_filtered.to_csv(index=False).encode(),
                file_name="model_leaderboard.csv",
                mime="text/csv"
            )
    else:
        st.info("No models have been trained yet. Run AutoML to populate leaderboard.")
