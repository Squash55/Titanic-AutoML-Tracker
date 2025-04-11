import streamlit as st
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tempfile

from tpot_connector import __dict__ as _tpot_cache


def run_ensemble_builder():
    st.title("🧬 Ensemble Builder (TPOT + RandomForest)")

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
        y_proba = ensemble.predict_proba(X_test)[:, 1] if hasattr(ensemble, "predict_proba") else None
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        _tpot_cache["latest_ensemble_model"] = ensemble

        st.success(f"✅ Ensemble accuracy: **{acc:.3f}**, F1: **{f1:.3f}**, Precision: **{prec:.3f}**, Recall: **{rec:.3f}**")

        st.markdown("### 📊 Sample Predictions")
        sample = pd.DataFrame({
            "Actual": y_test[:10].values,
            "Prediction": y_pred[:10]
        })
        st.dataframe(sample)

        # 📈 Probability Histogram
        if y_proba is not None:
            st.markdown("### 📈 Confidence Histogram")
            fig, ax = plt.subplots()
            sns.histplot(y_proba, bins=10, kde=True, ax=ax, color='skyblue')
            ax.set_title("Predicted Probability Distribution")
            st.pyplot(fig)

        # 💾 Download Button
        st.markdown("### 💾 Download Ensemble Model")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            joblib.dump(ensemble, tmp.name)
            with open(tmp.name, "rb") as f:
                st.download_button("📥 Download .pkl", data=f, file_name="ensemble_model.pkl")

        # 🧪 Add to leaderboard (optional storage stub)
        if st.button("📋 Add to Leaderboard"):
            _tpot_cache["leaderboard"] = _tpot_cache.get("leaderboard", []) + [{
                "Model": "VotingClassifier (TPOT + RF)",
                "Accuracy": acc,
                "F1": f1,
                "Precision": prec,
                "Recall": rec
            }]
            st.success("✅ Added to internal leaderboard cache.")

        # === AI Insights ===
        st.markdown("### 🧠 AI Insights")
        st.write("""
        **Ensemble Models** like the VotingClassifier can significantly improve prediction accuracy by combining the strengths of multiple models. In this case, the **TPOT** and **Random Forest** models bring complementary strengths:
        
        - **TPOT**: Automatically searches for the best pipeline and feature selection strategies.
        - **Random Forest**: Handles overfitting by averaging the results of many decision trees.

        When combined in an ensemble model, we leverage the **diversity** of the two models to boost overall predictive performance.

        **AI Insights for Improvement**:
        - **Hyperparameter Tuning**: Experiment with hyperparameter tuning on each model individually before building the ensemble. This can increase accuracy.
        - **Model Selection**: Adding more models to the ensemble (e.g., XGBoost, SVM) can further improve results, particularly for complex datasets.
        - **Class Imbalance**: For imbalanced datasets, consider adjusting the class weights for the Random Forest model.
        """)

    except Exception as e:
        st.error(f"❌ Ensemble prediction failed: {type(e).__name__}: {e}")
