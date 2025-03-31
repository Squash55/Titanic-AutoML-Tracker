
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import inspect

def get_hyperparams(model_class):
    """Get init parameters of a scikit-learn compatible model class."""
    sig = inspect.signature(model_class.__init__)
    params = {
        k: v.default for k, v in sig.parameters.items()
        if k != 'self' and v.default is not inspect.Parameter.empty
    }
    return params

def show_algo_selector():
    st.title("🧠 Algorithm Selector + HPO Panel")
    st.markdown("Explore all available classification models and tweak their hyperparameters before testing.")

    model_map = {
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "LogisticRegression": LogisticRegression,
        "SVC (Support Vector Machine)": SVC,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "GaussianNB (Naive Bayes)": GaussianNB,
        "KNeighborsClassifier": KNeighborsClassifier,
        "AdaBoostClassifier": AdaBoostClassifier,
        "XGBoostClassifier": XGBClassifier,
        "LightGBMClassifier": LGBMClassifier
    }

    model_name = st.selectbox("Select a Classification Model", list(model_map.keys()))
    ModelClass = model_map[model_name]

    st.subheader(f"⚙️ Hyperparameters for {model_name}")
    default_params = get_hyperparams(ModelClass)
    user_params = {}

    for param, default in default_params.items():
        if isinstance(default, bool):
            user_params[param] = st.checkbox(param, value=default)
        elif isinstance(default, int):
            user_params[param] = st.number_input(param, value=default)
        elif isinstance(default, float):
            user_params[param] = st.number_input(param, value=default, format="%.5f")
        elif isinstance(default, str):
            user_params[param] = st.text_input(param, value=default)
        else:
            user_params[param] = st.text_input(param, value=str(default))

    if st.button("✅ Instantiate Model"):
        try:
            model = ModelClass(**user_params)
            st.success(f"{model_name} initialized with selected hyperparameters!")
            st.code(str(model), language='python')
        except Exception as e:
            st.error(f"Error creating model: {e}")
