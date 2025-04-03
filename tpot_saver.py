
# tpot_saver.py

import streamlit as st
import joblib
import os

from tpot_connector import __dict__ as _tpot_cache

DEFAULT_PATH = "best_pipeline.pkl"

def save_tpot_pipeline(model, path=DEFAULT_PATH):
    joblib.dump(model, path)
    st.success(f"✅ TPOT pipeline saved to: {path}")

def load_tpot_pipeline(path=DEFAULT_PATH):
    if os.path.exists(path):
        model = joblib.load(path)
        _tpot_cache["latest_tpot_model"] = model
        st.success(f"✅ Loaded TPOT pipeline from: {path}")
        return model
    else:
        st.warning(f"⚠️ No saved pipeline found at {path}.")
        return None

def run_tpot_saver():
    st.subheader("💾 Saved Models")

    save_path = st.text_input("📁 File path to save/load TPOT model", value=DEFAULT_PATH)

    if _tpot_cache.get("latest_tpot_model") is not None:
        if st.button("📥 Save Current TPOT Model"):
            save_tpot_pipeline(_tpot_cache["latest_tpot_model"], path=save_path)

    if st.button("🔁 Load TPOT Model from File"):
        load_tpot_pipeline(path=save_path)
