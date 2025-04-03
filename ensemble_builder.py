
# ensemble_builder.py

import streamlit as st
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

from tpot_connector import __dict__ as _tpot_cache


def run_ensemble_builder():
    st.subheader("🧬 Ensemble Builder (TPOT + RandomForest)")

    st.write("✅ Ensemble Builder module is loading")  # Debug confirmation

    if "latest_tpot_model" not in _tpot_cache and "latest_rf_model" not in _tpot_cache:
        st.warning("⚠️ No models found in memory. Please run TPOT and RandomForest first.")
        return

    tpot_model = _tpot_cache.get("latest_tpot_model")
    rf_model = _tpot_cache.get("latest_rf_model")
    X_test = _tpot_cache.get("latest_X_test")
    y_test = _tpot_cache.get("latest_y_test")

    st.write(f"📦 TPOT loaded: {tpot_model is not None}")
    st.write(f"📦 RF loaded: {rf_model is not None}")
    st.write(f"📊 Test set available: {X_test is not None and y_test is not None}")

    if not tpot_model or not rf_model:
        st.warning("⚠️ Both TPOT and RandomForest models must be trained first.")
        return

    if X_test is None or y_test is None:
        st.warning("⚠️ Missing test data. Please run AutoML first.")
        return

    # Build soft voting ensemble
    ensemble = VotingClassifier(
        estimators=[("tpot", tpot_model), ("rf", rf_model)],
        voting="soft"
    )

    try:
        ensemble.fit(X_test, y_test)
    except Exception as e:
        st.info(f"ℹ️ Skipped ensemble.fit due to pipeline constraints: {type(e).__name__}: {e}")

    try:
        y_pred = ensemble.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Store the ensemble model
        _tpot_cache["latest_ensemble_model"] = ensemble

        st.success(f"✅ Ensemble accuracy on test set: **{acc:.3f}**")
        st.markdown("### 📊 Sample Predictions (first 10)")
        sample = pd.DataFrame({
            "Actual": y_test[:10].values,
            "Ensemble Prediction": y_pred[:10]
        })
        st.dataframe(sample)
    except Exception as e:
        st.error(f"❌ Ensemble prediction failed: {type(e).__name__}: {e}")
