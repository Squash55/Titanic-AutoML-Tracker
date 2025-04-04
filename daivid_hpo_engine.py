
# daivid_hpo_engine.py
import streamlit as st
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from tpot_connector import _tpot_cache

def run_daivid_hpo_engine():
    st.title("⚙️ DAIVID HPO Engine")
    st.markdown("This engine trains a model using your selected HPO configuration.")

    config = _tpot_cache.get("last_hpo_config")
    X = _tpot_cache.get("X_train")
    y = _tpot_cache.get("y_train")

    if config is None or X is None or y is None:
        st.warning("⚠️ Missing HPO config or training data. Please run the Smart HPO Recommender first.")
        return

    st.markdown("### 🔍 Configuration Summary")
    st.json(config)

    model_name = config["model"]
    test_size = config["test_size"] / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Neural Network":
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    else:
        st.error(f"Unsupported model: {model_name}")
        return

    with st.spinner("Training model..."):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, preds)
    st.success(f"✅ Accuracy: {acc:.4f}")
    if proba is not None:
        auc = roc_auc_score(y_test, proba)
        st.info(f"ROC AUC: {auc:.4f}")

    _tpot_cache["model"] = model
    _tpot_cache["X_test"] = X_test
    _tpot_cache["y_test"] = y_test
    _tpot_cache["y_pred"] = preds
    if proba is not None:
        _tpot_cache["y_pred_proba"] = proba

    st.markdown("✅ Model stored in session. You can now run SHAP, Golden Q&A, and Threshold Optimization.")
