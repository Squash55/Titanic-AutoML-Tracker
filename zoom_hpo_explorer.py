# zoom_hpo_explorer.py
import streamlit as st
import traceback
import optuna
import pandas as pd
import plotly.express as px
from tpot_connector import _tpot_cache
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_zoom_hpo_explorer():
    try:
        st.title("🔍 Zoomed HPO Explorer")
        st.markdown("""
        Explore deeper regions of the hyperparameter space through multiple zoom phases using Optuna.

        This method incrementally focuses on the most promising area of the search space — similar to microscope zooming.
        Each zoom phase builds on the prior's best region.

        ---
        🧠 **Why it matters:**  
        Sometimes broad HPO scans miss narrow pockets of optimal performance. Recursive zooming allows you to explore these.

        📚 _Inspired by concepts from Bayesian Optimization refinement and multi-resolution search strategies._
        """)

        X = _tpot_cache.get("X_train")
        y = _tpot_cache.get("y_train")

        if X is None or y is None:
            st.warning("⚠️ Please run AutoML first.")
            return

        zoom_levels = st.slider("Zoom Phases", 1, 5, 3)
        trials_per_zoom = st.slider("Trials per Zoom", 10, 100, 30)
        optimize_for = st.selectbox("Metric to Optimize", ["Accuracy"])
        run_parallel = st.checkbox("Enable Parallel Mode (Simulated)", value=True)

        zoom_summaries = []

        st.markdown("### 🚀 Running HPO Zoom Phases")
        current_bounds = {
            "n_estimators": (10, 200),
            "max_depth": (2, 20),
            "min_samples_split": (2, 10)
        }

        for zoom in range(zoom_levels):
            st.markdown(f"#### 🔎 Zoom Level {zoom+1}")

            def objective(trial):
                n_estimators = trial.suggest_int("n_estimators", *current_bounds["n_estimators"])
                max_depth = trial.suggest_int("max_depth", *current_bounds["max_depth"])
                min_samples_split = trial.suggest_int("min_samples_split", *current_bounds["min_samples_split"])

                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_val)
                return accuracy_score(y_val, preds)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=trials_per_zoom)

            best_params = study.best_params
            best_score = study.best_value
            st.success(f"Zoom {zoom+1} Best Score: {best_score:.4f}")
            st.code(best_params)

            # Visualize
            fig = px.line(
                x=list(range(len(study.trials))),
                y=[t.value for t in study.trials],
                labels={"x": "Trial", "y": "Score"},
                title=f"Zoom {zoom+1} Trial Scores"
            )
            st.plotly_chart(fig)

            zoom_summaries.append((zoom+1, best_score, best_params))

            # Refine bounds (±25% around best param)
            for k, v in best_params.items():
                low, high = current_bounds[k]
                span = high - low
                new_low = max(1, int(v - span * 0.25))
                new_high = max(new_low + 1, int(v + span * 0.25))
                current_bounds[k] = (new_low, new_high)

        st.markdown("### 📊 Final Zoom Summary")
        st.dataframe(pd.DataFrame(zoom_summaries, columns=["Zoom", "Score", "Params"]))

    except Exception as e:
        st.error(f"❌ Zoomed HPO Explorer failed to load: {type(e).__name__}: {e}")
        st.code(traceback.format_exc())
