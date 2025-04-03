# saved_models.py

import streamlit as st
import pickle
from tpot_connector import __dict__ as _tpot_cache


def run_saved_models_panel():
    st.subheader("💾 Saved Models Panel")

    models = {
        "TPOT Model": _tpot_cache.get("latest_tpot_model"),
        "RandomForest Model": _tpot_cache.get("latest_rf_model"),
        "Ensemble Model": _tpot_cache.get("latest_ensemble_model")
    }

    for name, model in models.items():
        if model:
            st.markdown(f"### 📦 {name}")
            model_bytes = pickle.dumps(model)
            st.download_button(
                label=f"⬇️ Download {name} (.pkl)",
                data=model_bytes,
                file_name=name.lower().replace(" ", "_") + ".pkl",
                mime="application/octet-stream"
            )
        else:
            st.markdown(f"### 📦 {name}")
            st.info(f"{name} not yet available. Train it first in AutoML or Ensemble panel.")