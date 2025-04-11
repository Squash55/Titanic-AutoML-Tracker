# automl_with_validation.py

import streamlit as st
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tpot import TPOTClassifier
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
from tpot_connector import _tpot_cache


# Helper function to get hyperparameters
def get_hyperparams(model_class):
    sig = inspect.signature(model_class.__init__)
    return {
        k: v.default for k, v in sig.parameters.items()
        if k != 'self' and v.default is not inspect.Parameter.empty
    }

# Function to generate synthetic scenarios for validation
def generate_synthetic_scenarios(X, num_cases=5):
    scenarios = []
    for _ in range(num_cases):
        scenario = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                scenario[col] = round(random.uniform(X[col].min(), X[col].max()), 2)
            else:
                scenario[col] = random.choice(X[col].dropna().unique())
        scenarios.append(scenario)
    return pd.DataFrame(scenarios)

# Function for launching AutoML (TPOT)
@st.cache_data
def load_titanic_data():
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    df = df.fillna(df.median(numeric_only=True))
    df = df.dropna()
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Main run function
def run():
    st.title("🚀 AutoML Pipeline with Model Selection and Validation Scenarios")
# Purpose statement
    st.markdown("""
    This tool allows you to select an algorithm, fine-tune hyperparameters, and run AutoML using TPOT. 
    It also generates synthetic validation scenarios for robustness testing, helping to evaluate model performance on edge cases.
    """)
    # Step 1: Algorithm Selector
    task_type = st.radio("Select Task Type", ["Classification", "Regression"])
    
    models = {}
    if task_type == "Classification":
        models = {
            "LogisticRegression": LogisticRegression,
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
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "KNeighborsRegressor": KNeighborsRegressor,
            "SVR": SVR,
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

    # Step 2: AutoML Launcher (TPOT Demo)
    st.subheader("🚢 Titanic AutoML Launcher (TPOT Demo)")

    st.info("Running a real TPOT model on the Titanic dataset...")
    X_train, X_test, y_train, y_test = load_titanic_data()

    tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, max_time_mins=2, random_state=42)
    with st.spinner("⏳ TPOT is optimizing models..."):
        tpot.fit(X_train, y_train)

    y_pred = tpot.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"✅ TPOT Finished. Accuracy on Test Set: **{acc:.3f}**")
    st.markdown("### 📜 Best Pipeline Code")
    st.code(tpot.export(), language="python")

    st.markdown("### 🧪 Predictions Sample")
    sample = pd.DataFrame({"Actual": y_test.values[:10], "Predicted": y_pred[:10]})
    st.dataframe(sample)

    # Step 3: AI-Generated Validation Scenarios
    st.title("🧪 AI-Generated Validation Scenarios")

    model = _tpot_cache.get("latest_tpot_model")
    X_train = _tpot_cache.get("latest_X_train")

    if model is None or X_train is None:
        st.warning("⚠️ TPOT model or training data not found. Please run AutoML first.")
        return

    num_cases = st.slider("How many test cases to generate?", 3, 10, 5)
    X_synthetic = generate_synthetic_scenarios(X_train, num_cases=num_cases)

    st.markdown("### 🧪 Generated Scenarios")
    st.dataframe(X_synthetic)

    if st.button("⚙️ Predict on These Scenarios"):
        try:
            preds = model.predict(X_synthetic)
            X_synthetic["Prediction"] = preds
            st.success("✅ Model predictions computed.")
            st.dataframe(X_synthetic)
        except Exception as e:
            st.error(f"❌ Prediction failed: {type(e).__name__}: {e}")
