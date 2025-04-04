
# daivid_hpo_engine.py
import streamlit as st
import optuna
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from tpot_connector import _tpot_cache

def run_daivid_hpo_engine():
    st.title("⚙️ DAIVID HPO Engine")

    cfg = _tpot_cache.get("last_hpo_config")
    X = _tpot_cache.get("X_train")
    y = _tpot_cache.get("y_train")

    if cfg is None or X is None or y is None:
        st.error("❌ Configuration or training data not found. Please run AutoML and Smart Recommender first.")
        return

    st.markdown("### ⚗️ HPO Config Overview")
    st.json(cfg)

    test_size = cfg["test_size"] / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    st.markdown("### 📊 Live Training Progress")
    progress = st.progress(0)

    model_name = cfg["model"]
    def get_model(trial):
        if model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10)
            )
        elif model_name == "Logistic Regression":
            return LogisticRegression(
                C=trial.suggest_float("C", 0.001, 10.0, log=True),
                penalty="l2",
                solver="lbfgs"
            )
        elif model_name == "XGBoost":
            return XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                use_label_encoder=False,
                eval_metric="logloss"
            )
        elif model_name == "Neural Network":
            return MLPClassifier(
                hidden_layer_sizes=(trial.suggest_int("layer1", 32, 128), trial.suggest_int("layer2", 16, 64)),
                activation="relu",
                solver="adam",
                max_iter=200
            )

    norm = cfg["norm"]
    if norm == "MinMax":
        scaler = MinMaxScaler()
    elif norm == "Z-Score":
        scaler = StandardScaler()
    elif norm == "Robust":
        scaler = RobustScaler()
    else:
        scaler = None

    def objective(trial):
        model = get_model(trial)
        pipe_steps = []
        if scaler:
            pipe_steps.append(("scaler", scaler))
        pipe_steps.append(("model", model))
        pipeline = Pipeline(pipe_steps)

        if cfg["calibrated"]:
            pipeline = CalibratedClassifierCV(pipeline)

        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg["max_models"], n_jobs=-1 if cfg["parallel"] else 1)

    st.success(f"✅ Best trial: {study.best_trial.value:.4f}")
    best_model = get_model(study.best_trial)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    st.markdown("### 🔍 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.markdown("### 📈 Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.markdown("### 🔬 SHAP Interpretability")
    try:
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_train)
        fig2 = shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(bbox_inches='tight')
    except:
        st.warning("SHAP not available for this model.")

    _tpot_cache["daivid_model"] = best_model
    _tpot_cache["y_test"] = y_test
    _tpot_cache["y_pred_proba"] = y_proba

    st.markdown("### ✅ Next Steps")
    st.success("Model and SHAP values cached. You may now go to Threshold Optimizer or SHAP Panel.")
    if st.button("Go to SHAP Panel"):
        st.session_state.tab = "SHAP Panel"
    if st.button("Go to Threshold Optimizer"):
        st.session_state.tab = "Threshold Optimizer"
