
# automl_comparison.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tpot import TPOTClassifier
import time


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


def run_automl_comparison():
    st.subheader("🤖 AutoML Comparison Panel")

    X_train, X_test, y_train, y_test = load_titanic_data()

    results = []

    # TPOT AutoML
    if st.button("⚙️ Run TPOT AutoML"):
        start = time.time()
        tpot = TPOTClassifier(generations=3, population_size=10, verbosity=0, max_time_mins=2, random_state=42)
        tpot.fit(X_train, y_train)
        acc = accuracy_score(y_test, tpot.predict(X_test))
        end = time.time()
        results.append(("TPOT", acc, end - start))
        st.success(f"TPOT accuracy: {acc:.3f}, time: {end - start:.1f}s")

    # Random Forest Baseline
    if st.button("🌲 Run RandomForest Baseline"):
        _tpot_cache["latest_rf_model"] = model
        start = time.time()
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        end = time.time()
        results.append(("RandomForest", acc, end - start))
        st.success(f"RandomForest accuracy: {acc:.3f}, time: {end - start:.1f}s")

    # Show results table + bar chart
    if results:
        df = pd.DataFrame(results, columns=["Model", "Accuracy", "Training Time (s)"])
        st.markdown("### 📊 Comparison Results")
        st.dataframe(df)
        st.bar_chart(df.set_index("Model")[["Accuracy", "Training Time (s)"]])
    else:
        st.info("Click a button above to run and compare AutoML tools.")
