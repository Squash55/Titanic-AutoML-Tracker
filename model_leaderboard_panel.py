# model_leaderboard_panel.py

import streamlit as st
import pandas as pd
from tpot_connector import _tpot_cache


def run_model_leaderboard_panel():
    st.title("🏆 Model Leaderboard Tracker")

    models = _tpot_cache.get("all_models", {})
    X_test = _tpot_cache.get("X_test")
    y_test = _tpot_cache.get("y_test")

    rows = []
    for name, model in models.items():
        acc = "-"
        try:
            if hasattr(model, "predict") and X_test is not None and y_test is not None:
                if hasattr(model, "leaderboard"):
                    preds = model.predict(X_test)
                    acc = (preds == y_test).mean()
                else:
                    acc = model.score(X_test, y_test)
        except:
            pass
        rows.append({"Model Name": name, "Type": type(model).__name__, "Accuracy": acc})

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        if st.button("📥 Export Leaderboard to CSV"):
            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False).encode(),
                file_name="model_leaderboard.csv",
                mime="text/csv"
            )
    else:
        st.info("No models have been trained yet. Run AutoML to populate leaderboard.")
