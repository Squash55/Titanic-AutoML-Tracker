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
    st.markdown("""
    This engine trains a model using your selected HPO configuration. Hyperparameter optimization (HPO) helps in finding the most optimal model parameters to improve model performance.

    The selected HPO config will determine which model is trained and how it is tuned.
    """)

    # Load the HPO configuration and training data
    config = _tpot_cache.get("last_hpo_config")
    X = _tpot_cache.get("X_train")
    y = _tpot_cache.get("y_train")

    if config is None or X is None or y is None:
        st.warning("⚠️ Missing HPO config or training data. Please run the Smart HPO Recommender first.")
        return

    # Display configuration summary
    st.markdown("### 🔍 Configuration Summary")
    st.json(config)

    model_name = config["model"]
    test_size = config["test_size"] / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Model selection based on config
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        st.markdown("### Model: Random Forest")
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        st.markdown("### Model: Logistic Regression")
    elif model_name == "Neural Network":
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
        st.markdown("### Model: Neural Network")
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        st.markdown("### Model: XGBoost")
    else:
        st.error(f"Unsupported model: {model_name}")
        return

    # Hyperparameter tuning - Display the tuned parameters if available
    st.markdown("### 🔧 HPO Parameters Used")
    st.write(config.get("hyperparameters", "No hyperparameters provided"))

    # Training the model
    with st.spinner("Training model..."):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Evaluation metrics
    acc = accuracy_score(y_test, preds)
    st.success(f"✅ Accuracy: {acc:.4f}")
    
    if proba is not None:
        auc = roc_auc_score(y_test, proba)
        st.info(f"ROC AUC: {auc:.4f}")

    # AI Insights: Performance comparison and interpretation
    st.markdown("### 🧠 AI Insights")

    st.write(f"Model: {model_name}")
    st.write(f"Accuracy: {acc:.4f}")
    if proba is not None:
        st.write(f"ROC AUC: {auc:.4f}")

    # Interpretation based on model type
    if model_name == "Random Forest":
        st.write("Random Forest works well for capturing non-linear relationships and feature importance. It is less prone to overfitting compared to many other models.")
    elif model_name == "Logistic Regression":
        st.write("Logistic Regression is a simple model and works best when the relationship between features is linear. It is often used for classification problems.")
    elif model_name == "Neural Network":
        st.write("Neural Networks can model highly complex relationships in the data, but they require careful tuning to avoid overfitting, especially with small datasets.")
    elif model_name == "XGBoost":
        st.write("XGBoost is a powerful model that performs well on a variety of tasks, especially with large datasets. It is prone to overfitting if not carefully tuned.")

    # Store the model and its predictions in session for further use
    _tpot_cache["model"] = model
    _tpot_cache["X_test"] = X_test
    _tpot_cache["y_test"] = y_test
    _tpot_cache["y_pred"] = preds
    if proba is not None:
        _tpot_cache["y_pred_proba"] = proba

    st.markdown("✅ Model stored in session. You can now run SHAP, Golden Q&A, and Threshold Optimization.")
