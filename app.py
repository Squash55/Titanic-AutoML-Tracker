import os
print("Current files:", os.listdir())

import streamlit as st
import traceback  # <-- move this right after streamlit

st.set_page_config(page_title="Titanic AutoML App", layout="wide")

from tpot_connector import _tpot_cache
# -- Distribution Auditor Safe Import --
try:
    from distribution_auditor import run_distribution_auditor
except ImportError:
    def run_distribution_auditor():
        st.error("❌ Distribution Auditor failed to load. Check for typos or missing file.")

try:
    from smart_hpo_recommender import run_smart_hpo_recommender
except ImportError:
    def run_smart_hpo_recommender():
        st.error("❌ Smart HPO Recommender failed to load. Ensure smart_hpo_recommender.py exists and is error-free.")

# -- DAIVID HPO Trainer Safe Import --
try:
    from daivid_hpo_trainer import run_daivid_hpo_trainer
except Exception as e:
    import traceback
    st.error(f"❌ DAIVID HPO Trainer failed to import: {type(e).__name__}: {e}")
    st.code(traceback.format_exc())
    def run_daivid_hpo_trainer():
        st.error("❌ DAIVID HPO Trainer is unavailable due to earlier import error.")

# -- DAIVID HPO Engine Safe Import --
try:
    from daivid_hpo_engine import run_daivid_hpo_engine
except ImportError:
    def run_daivid_hpo_engine():
        st.error("❌ DAIVID HPO Engine failed to load. Ensure daivid_hpo_engine.py exists and is error-free.")

# -- SHAP Comparison Panel Safe Import --
try:
    from shap_comparison import run_shap_comparison
except ImportError:
    def run_shap_comparison():
        st.error("❌ SHAP Comparison panel failed to load. Ensure shap_comparison.py is present.")
# -- AutoML Comparison Safe Import --
try:
    from automl_comparison import run_automl_comparison
except ImportError:
    def run_automl_comparison():
        st.error("❌ AutoML Comparison panel failed to load. Ensure automl_comparison.py exists.")

# -- Threshold Optimizer Safe Import --
try:
    from threshold_optimizer import run_threshold_optimizer
except ImportError:
    def run_threshold_optimizer(y_true=None, y_proba=None):
        st.error("❌ Threshold Optimizer failed to load. Check threshold_optimizer.py is present and error-free.")
# -- DOE Panel Safe Import --
try:
    from doe_panel import run_doe_panel
except ImportError:
    def run_doe_panel(df=None, model=None):
        st.error("❌ DOE Panel failed to load. Ensure doe_panel.py exists and is error-free.")
# -- Safely import SHAP Panel --
try:
    from shap_interpretability import run_shap_panel
except ImportError:
    def run_shap_panel():
        st.error("❌ SHAP Panel failed to load. Check shap_interpretability.py is present and correctly named.")
        
try:
    from shap_waterfall import run_shap_waterfall
except ImportError:
    def run_shap_waterfall():
        st.error("❌ SHAP Waterfall Panel failed to load.")

# Near the top
try:
    from auto_eda import run_auto_eda
except Exception:
    import traceback
    def run_auto_eda():
        st.error("❌ Auto EDA panel failed to load.")
        st.code(traceback.format_exc())
        
try:
    from pdf_report import run_pdf_report
except ImportError:
    def run_pdf_report():
        st.error("❌ PDF Report module failed to load.")
        
try:
    from auto_threshold_optimizer import run_threshold_optimizer
except:
    def run_threshold_optimizer():
        st.error("❌ Threshold Optimization panel failed to load.")

try:
    from tpot_saver import run_tpot_saver
except ImportError:
    def run_tpot_saver():
        st.error("❌ Saved Models panel failed to load.")
        
try:
    from experiment_tracker import run_experiment_tracker
except:
    def run_experiment_tracker():
        st.error("❌ Experiment Tracker panel failed to load.")

try:
    from automl_comparison import run_automl_comparison
except Exception as e:
    import streamlit as st
    st.error(f"❌ AutoML Comparison panel failed to load: {type(e).__name__}: {e}")
try:
    from shap_comparison import run_shap_comparison
except:
    def run_shap_comparison():
        st.error("❌ SHAP Comparison Panel failed to load.")

try:
    from shap_comparison import run_shap_comparison
except:
    def run_shap_comparison():
        st.error("❌ SHAP Comparison panel failed to load.")
try:
    from ensemble_builder import run_ensemble_builder
except:
    def run_ensemble_builder():
        st.error("❌ Ensemble Builder failed to load.")

# Notebook Scout
try:
    from notebook_scout import run_notebook_scout
except Exception as e:
    def run_notebook_scout():
        st.error(f"❌ Notebook Scout failed to load. Error: {type(e).__name__}: {e}")

# -- Import Golden Q&A --
try:
    from golden_qa import run_golden_qa
except ImportError:
    def run_golden_qa():
        st.error("❌ Golden Q&A Panel failed to load. Check golden_qa.py is present and correctly named.")
# automl launcher
try:
    from automl_launcher import run_automl_launcher
except ImportError:
    def run_automl_launcher():
        st.error("❌ AutoML Launcher failed to load. Check automl_launcher.py is present.")

try:
    from saved_models import run_saved_models_panel
except:
    def run_saved_models_panel():
        st.error("❌ Saved Models Panel failed to load.")

# -- Sidebar navigation --
st.sidebar.title("📊 Navigation")
tab = st.sidebar.radio("Choose a Tab:", [
    "AutoML Launcher", "Algorithm Selector", "Golden Q&A", "SHAP Panel",
    "Notebook Scout", "SHAP Waterfall", "PDF Report", "Saved Models", "AutoML Launcher", "Algorithm Selector", "AutoML Comparison", "SHAP Comparison", "Ensemble Builder", "SHAP Comparison", "Experiment Tracker", "Threshold Optimizer", "Saved Models", "Auto EDA",  "DOE Panel", "Threshold Optimizer", "AutoML Comparison", "SHAP Comparison", "Distribution Auditor", "Smart HPO Recommender", "DAIVID HPO Engine",
 "DAIVID HPO Trainer", "Zoomed HPO Explorer"

])


# -- Tab 1: AutoML Launcher --
def run_automl_launcher():
    st.subheader("🚢 Titanic AutoML Launcher")
    automl_tool = st.selectbox("Choose AutoML Tool", ["TPOT", "H2O.ai"])
    parallel_mode = st.checkbox("🔁 Run in Parallel (Dask mode)", value=False)
    if st.button("🚀 Launch AutoML"):
        with st.spinner("Running AutoML..."):
            st.success(f"{automl_tool} run completed!")
            st.code("This is placeholder output. Real model training will be added next.")

# -- Tab 2: Algorithm Selector --
def run_algorithm_selector():
    st.subheader("🧠 Algorithm Selector (Dual Mode)")
    mode = st.radio("Select Mode:", ["Classification", "Regression"], horizontal=True)

    if mode == "Classification":
        algorithms = {
            "Logistic Regression": {
                "Pros": "Interpretable, fast.",
                "Cons": "Assumes linearity, sensitive to outliers.",
                "When to Use": "Simple binary problems, baseline models."
            },
            "Random Forest": {
                "Pros": "High accuracy, robust.",
                "Cons": "Less interpretable.",
                "When to Use": "General-purpose tabular classification."
            },
            "XGBoost": {
                "Pros": "Top performer in competitions.",
                "Cons": "Requires tuning, less interpretable.",
                "When to Use": "Accuracy-critical tabular problems."
            },
            "CatBoost": {
                "Pros": "Handles categorical data natively.",
                "Cons": "Newer, fewer tutorials.",
                "When to Use": "Mixed-type data with categorical features."
            },
            "TabNet": {
                "Pros": "Deep learning for tabular data.",
                "Cons": "Heavier training, less intuitive.",
                "When to Use": "When capturing complex patterns is critical."
            }
        }
    else:
        algorithms = {
            "Linear Regression": {
                "Pros": "Simple, interpretable.",
                "Cons": "Assumes linear relationships.",
                "When to Use": "Fast baseline or simple trends."
            },
            "Random Forest Regressor": {
                "Pros": "Accurate, handles non-linearity.",
                "Cons": "Can overfit, less interpretable.",
                "When to Use": "Flexible, general-purpose regression."
            },
            "XGBoost Regressor": {
                "Pros": "Great accuracy, powerful.",
                "Cons": "Requires tuning.",
                "When to Use": "When performance is key."
            },
            "ExtraTrees Regressor": {
                "Pros": "Very fast, robust.",
                "Cons": "Can be noisy.",
                "When to Use": "Fast ensemble alternative to Random Forest."
            },
            "MLP Regressor": {
                "Pros": "Can model complex patterns.",
                "Cons": "Needs tuning and data prep.",
                "When to Use": "Deep relationships in features."
            }
        }

    selected_algo = st.selectbox("Choose an Algorithm:", list(algorithms.keys()))
    info = algorithms[selected_algo]
    st.markdown(f"### ℹ️ {selected_algo} Info")
    st.markdown(f"**Pros:** {info['Pros']}")
    st.markdown(f"**Cons:** {info['Cons']}")
    st.markdown(f"**When to Use:** {info['When to Use']}")

# -- Tab Routing --
if tab == "AutoML Launcher":
    run_automl_launcher()
elif tab == "Algorithm Selector":
    run_algorithm_selector()
elif tab == "Golden Q&A":
    run_golden_qa()
elif tab == "SHAP Panel":
    run_shap_panel()
elif tab == "Notebook Scout":
    run_notebook_scout()
elif tab == "SHAP Waterfall":
    run_shap_waterfall()
elif tab == "PDF Report":
    run_pdf_report()
elif tab == "Saved Models":
    run_tpot_saver()
elif tab == "AutoML Comparison":
    run_automl_comparison()
elif tab == "Ensemble Builder":
    run_ensemble_builder()
elif tab == "SHAP Comparison":
    run_shap_comparison()
elif tab == "Saved Models":
    run_saved_models_panel()
elif tab == "Auto EDA":
    run_auto_eda()
elif tab == "DOE Panel":
    if "X_train" in st.session_state and "model" in st.session_state:
        run_doe_panel(df=st.session_state["X_train"], model=st.session_state["model"])
    else:
        st.warning("🚧 Required objects missing. Train a model first to use the DOE panel.")
# -- Threshold Optimizer Tab --
elif tab == "Threshold Optimizer":
    from tpot_connector import _tpot_cache  # 🧠 This must exist earlier in your project
    y_true = _tpot_cache.get("y_test")
    y_proba = _tpot_cache.get("y_pred_proba")

    if y_true is not None and y_proba is not None:
        run_threshold_optimizer(y_true=y_true, y_proba=y_proba)
    else:
        st.warning("🟡 TPOT predictions not found. Please run AutoML first.")

# -- Zoomed HPO Trainer Safe Import --
try:
    from daivid_hpo_trainer import run_daivid_hpo_trainer
except Exception as e:
    import streamlit as st
    def run_daivid_hpo_trainer():
        st.error(f"❌ DAIVID HPO Trainer failed to load: {type(e).__name__}: {e}")
# -- Zoomed HPO Explorer Safe Import --
try:
    from zoom_hpo_explorer import run_zoom_hpo_explorer
except Exception as e:
    import streamlit as st
    error_message = f"❌ Zoomed HPO Explorer failed to load: {type(e).__name__}: {e}"
    def run_zoom_hpo_explorer():
        st.error(error_message)

# elif
# Sidebar Tab Routing
if tab == "AutoML Launcher":
    run_automl_launcher()
elif tab == "Algorithm Selector":
    run_algorithm_selector()
elif tab == "Threshold Optimizer":
    y_true = _tpot_cache.get("y_test")
    y_proba = _tpot_cache.get("y_pred_proba")
    if y_true is not None and y_proba is not None:
        run_threshold_optimizer(y_true=y_true, y_proba=y_proba)
    else:
        st.warning("🟡 TPOT predictions not found. Please run AutoML first.")
elif tab == "SHAP Comparison":
    run_shap_comparison()
elif tab == "Smart HPO Recommender":
    run_smart_hpo_recommender()
elif tab == "Distribution Auditor":
    run_distribution_auditor()
elif tab == "DAIVID HPO Engine":
    run_daivid_hpo_engine()
elif tab == "DAIVID HPO Trainer":
    run_daivid_hpo_trainer()
elif tab == "Zoomed HPO Explorer":
    run_zoom_hpo_explorer()







