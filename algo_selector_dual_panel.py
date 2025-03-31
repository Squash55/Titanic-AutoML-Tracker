
import streamlit as st
import inspect
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

def get_hyperparams(model_class):
    sig = inspect.signature(model_class.__init__)
    return {
        k: v.default for k, v in sig.parameters.items()
        if k != 'self' and v.default is not inspect.Parameter.empty
    }

def show_algo_selector():
    st.title("🧠 Algorithm Selector + HPO Panel")
    st.markdown("Select a task type, then pick a model and tune its hyperparameters.")

    task_type = st.radio("Select Task Type", ["Classification", "Regression"])

    models = {}
    if task_type == "Classification":
        models = {
            "LogisticRegression": LogisticRegression,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "KNeighborsClassifier": KNeighborsClassifier,
            "SVC (SVM Classifier)": SVC,
            "GaussianNB": GaussianNB,
            "MLPClassifier": MLPClassifier,
            "XGBClassifier": XGBClassifier,
            "LGBMClassifier": LGBMClassifier
        }
    else:
        models = {
            "LinearRegression": LinearRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "KNeighborsRegressor": KNeighborsRegressor,
            "SVR (SVM Regressor)": SVR,
            "MLPRegressor": MLPRegressor,
            "XGBRegressor": XGBRegressor,
            "LGBMRegressor": LGBMRegressor
        }

    model_name = st.selectbox("Choose a model", list(models.keys()))
    ModelClass = models[model_name]
    st.subheader(f"🔧 Hyperparameters for {model_name}")
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
            st.success(f"{model_name} created successfully!")
            st.code(str(model), language="python")
        except Exception as e:
            st.error(f"Error creating model: {e}")
