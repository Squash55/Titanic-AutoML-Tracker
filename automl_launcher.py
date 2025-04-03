# automl_launcher.py

import streamlit as st
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


@st.cache_data
def load_titanic_data():
    # Simulated Titanic dataset
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    df = df.fillna(df.median(numeric_only=True))
    df = df.dropna()
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def run_automl_launcher():
    st.subheader("🚢 Titanic AutoML Launcher (TPOT Demo)")

    st.info("Running a real TPOT model on the Titanic dataset...")
    X_train, X_test, y_train, y_test = load_titanic_data()

    tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, max_time_mins=2, random_state=42)
    with st.spinner("⏳ TPOT is optimizing models..."):
        tpot.fit(X_train, y_train)

    # 🔁 Store model and data for SHAP/Q&A modules
    from tpot_connector import (
        __dict__ as _tpot_cache
    )
    _tpot_cache["latest_tpot_model"] = tpot.fitted_pipeline_
    _tpot_cache["latest_X_train"] = X_train
    _tpot_cache["latest_y_train"] = y_train
    _tpot_cache["latest_X_test"] = X_test
    _tpot_cache["latest_y_test"] = y_test

    y_pred = tpot.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"✅ TPOT Finished. Accuracy on Test Set: **{acc:.3f}**")
    st.markdown("### 📜 Best Pipeline Code")
    st.code(tpot.export(), language="python")

    st.markdown("### 🧪 Predictions Sample")
    sample = pd.DataFrame({"Actual": y_test.values[:10], "Predicted": y_pred[:10]})
    st.dataframe(sample)
