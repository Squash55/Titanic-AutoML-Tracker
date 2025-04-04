# daivid_hpo_engine.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from tpot_connector import _tpot_cache


def run_daivid_hpo_engine():
    st.subheader("🧪 DAIVID Auto-HPO Engine")

    config = _tpot_cache.get("last_hpo_config")
    df = _tpot_cache.get("X_train")
    y = _tpot_cache.get("y_train")

    if not config or df is None or y is None:
        st.error("Missing config or training data. Please run Smart HPO Recommender first.")
        return

    st.info(f"Running HPO for: {config['model']}")

    # 1. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=config['test_size'] / 100.0, random_state=42
    )

    # 2. Normalization
    if config['norm'] == "Z-Score":
        scaler = StandardScaler()
    elif config['norm'] == "MinMax":
        scaler = MinMaxScaler()
    elif config['norm'] == "Robust":
        scaler = RobustScaler()
    else:
        scaler = None

    # 3. Model Selection
    base_model = None
    if config['model'] == "Random Forest":
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif config['model'] == "Logistic Regression":
        base_model = LogisticRegression(max_iter=500)
    elif config['model'] == "XGBoost":
        base_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    if config['calibrated']:
        base_model = CalibratedClassifierCV(base_model)

    # 4. Build pipeline
    steps = []
    if scaler:
        steps.append(('scaler', scaler))
    steps.append(('model', base_model))
    pipe = Pipeline(steps)

    # 5. Fit model
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, 'predict_proba') else None

    # 6. Score
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.success(f"Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

    # 7. Store to cache
    _tpot_cache['y_test'] = y_test
    _tpot_cache['y_pred'] = y_pred
    _tpot_cache['y_pred_proba'] = y_proba

    _tpot_cache.setdefault("all_models", {})[f"DAIVID_{config['model']}"] = pipe

    st.success(f"Model '{config['model']}' stored in DAIVID model zoo ✅")
