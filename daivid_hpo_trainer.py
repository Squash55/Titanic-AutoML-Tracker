import streamlit as st
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, make_scorer
from tpot_connector import _tpot_cache
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

def run_daivid_hpo_trainer():
    try:
        st.title("🧪 DAIVID HPO Trainer")
        st.markdown("""
        This module performs hyperparameter optimization using the selected configuration from the Smart HPO panel.

        Hyperparameter optimization (HPO) tunes the model by selecting the best hyperparameters to maximize performance. Optuna is used to evaluate various configurations and select the best-performing one.

        The purpose of this tool is to improve the accuracy and robustness of the model by optimizing parameters.
        """)

        config = _tpot_cache.get("last_hpo_config")
        X = _tpot_cache.get("X_train")
        y = _tpot_cache.get("y_train")

        if config is None or X is None or y is None:
            st.warning("⚠️ Training data not found. Please run AutoML first.")
            return

        st.markdown("### 🔍 Configuration Summary")
        st.json(config)

        model_choice = config["model"]

        def objective(trial):
            # Based on the chosen model, hyperparameters are tuned.
            if model_choice == "Random Forest":
                n_estimators = trial.suggest_int("n_estimators", 50, 300)
                max_depth = trial.suggest_int("max_depth", 3, 20)
                clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            elif model_choice == "XGBoost":
                n_estimators = trial.suggest_int("n_estimators", 50, 300)
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
                max_depth = trial.suggest_int("max_depth", 3, 10)
                clf = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
            elif model_choice == "Logistic Regression":
                C = trial.suggest_float("C", 0.01, 10.0, log=True)
                clf = LogisticRegression(C=C, max_iter=1000)
            else:
                raise ValueError(f"Unsupported model: {model_choice}")

            return cross_val_score(clf, X, y, scoring="accuracy", cv=3).mean()

        st.info("🔍 Running Optuna study...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=config.get("max_models", 10))

        st.success("✅ HPO Completed")
        st.write("Best Score:", study.best_value)
        st.write("Best Parameters:", study.best_params)

    except Exception as e:
        import traceback
        st.error(f"❌ DAIVID HPO Trainer failed to run: {type(e).__name__}: {e}")
        st.code(traceback.format_exc())

    # Allow user to select scoring metric
    scoring_map = {
        "AUC": (roc_auc_score, "roc_auc"),
        "Accuracy": (accuracy_score, "accuracy"),
        "F1": (f1_score, "f1")
    }
    scoring_label = st.selectbox("Evaluation Metric", list(scoring_map.keys()), index=0)
    score_func, score_name = scoring_map[scoring_label]

    def get_model(model_name, trial):
        if model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 10, 300),
                max_depth=trial.suggest_int("max_depth", 2, 20),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                random_state=42
            )
        elif model_name == "Logistic Regression":
            return LogisticRegression(
                C=trial.suggest_loguniform("C", 1e-4, 10),
                solver=trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
                max_iter=500,
                random_state=42
            )
        elif model_name == "XGBoost":
            return XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 2, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42
            )
        elif model_name == "Neural Network":
            return MLPClassifier(
                hidden_layer_sizes=trial.suggest_categorical("hidden_layer_sizes", [(64,), (128,), (64, 32), (128, 64)]),
                alpha=trial.suggest_loguniform("alpha", 1e-5, 1e-1),
                learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-1),
                max_iter=300,
                early_stopping=config.get("early_stopping", True),
                random_state=42
            )
        else:
            raise ValueError("Unknown model")

    def objective(trial):
        model = get_model(config["model"], trial)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val)
        preds_label = (preds > 0.5).astype(int) if preds.ndim > 1 else preds
        return score_func(y_val, preds_label)

    with st.spinner("🔄 Optimizing model using Optuna..."):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=config.get("max_models", 10))

    st.success("✅ Optimization Complete")
    st.write("Best Parameters:")
    st.json(study.best_params)
    st.metric("Best Score", f"{study.best_value:.4f} ({scoring_label})")

    # Refit best model on full data
    final_model = get_model(config["model"], study.best_trial)
    final_model.fit(X, y)
    _tpot_cache["best_model"] = final_model
    st.success("📦 Best model saved to cache. Ready for SHAP, Thresholding, or PDF Export.")
