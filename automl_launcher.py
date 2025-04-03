import streamlit as st
import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from tpot_connector import _tpot_cache


def run_automl_launcher():
    st.title("🚀 TPOT AutoML Launcher")

    uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Data Preview:", df.head())

        target_column = st.selectbox("Select Target Column", df.columns)
        if target_column:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            if st.button("▶️ Run TPOT AutoML"):
                with st.spinner("Training TPOT Classifier... this may take a minute..."):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
                    tpot.fit(X_train, y_train)

                    best_model = tpot.fitted_pipeline_

                    # ✅ Store in shared memory so other modules (Auto EDA) can access it
                    _tpot_cache["latest_X_train"] = X_train
                    _tpot_cache["latest_y_train"] = y_train
                    _tpot_cache["latest_tpot_model"] = best_model

                    st.success("✅ TPOT training complete!")
                    st.code(best_model)
    else:
        st.info("👆 Upload a CSV file to get started.")
