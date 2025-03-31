
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from autofeat import AutoFeatRegressor
import featuretools as ft

def show_autofe_playground():
    st.title("🧪 Feature Engineering Playground")
    st.markdown("Try different feature engineering techniques and preview results.")

    uploaded = st.file_uploader("Upload your Titanic training dataset (CSV)", type=["csv"], key="feupload")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("📊 Raw Data Preview")
        st.dataframe(df.head())

        method = st.selectbox("Choose a Feature Engineering Strategy", [
            "Raw",
            "Manual (Title, IsAlone, CabinKnown)",
            "Autofeat (Polynomial/Interaction Features)",
            "Featuretools (Deep Feature Synthesis)"
        ])

        if method == "Raw":
            st.info("Showing original dataset.")
            fe_df = df.copy()

        elif method == "Manual (Title, IsAlone, CabinKnown)":
            fe_df = df.copy()
            fe_df['Title'] = fe_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            fe_df['IsAlone'] = ((fe_df['SibSp'] + fe_df['Parch']) == 0).astype(int)
            fe_df['CabinKnown'] = fe_df['Cabin'].notnull().astype(int)
            st.success("Manual features added.")

        elif method == "Autofeat (Polynomial/Interaction Features)":
            df_num = df.select_dtypes(include=np.number).drop(columns=["PassengerId", "Survived"], errors='ignore')
            model = AutoFeatRegressor(verbose=0)
            X_transformed = model.fit_transform(df_num.values, df_num.columns)
            fe_df = pd.DataFrame(X_transformed, columns=model.new_feature_names_)
            st.success("Autofeat features generated.")

        elif method == "Featuretools (Deep Feature Synthesis)":
            try:
                df['PassengerId'] = df['PassengerId'].astype(int)
                es = ft.EntitySet(id="titanic")
                es = es.add_dataframe(dataframe_name="passengers", dataframe=df, index="PassengerId")
                fe_df, _ = ft.dfs(entityset=es, target_dataframe_name="passengers", max_depth=1)
                st.success("Featuretools deep features generated.")
            except Exception as e:
                st.error(f"Featuretools failed: {e}")
                return

        st.subheader("🔍 Transformed Features")
        st.dataframe(fe_df.head())
        st.download_button("Download Features CSV", data=fe_df.to_csv(index=False), file_name="engineered_features.csv")
