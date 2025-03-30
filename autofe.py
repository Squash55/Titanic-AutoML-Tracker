
import streamlit as st
import pandas as pd
import numpy as np
from autofeat import AutoFeatRegressor
import featuretools as ft

# --- Manual Feature Engineering Functions ---
def manual_engineering(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 120], labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])
    df['CabinKnown'] = df['Cabin'].notnull().astype(int)
    return df

# --- Autofeat Feature Engineering ---
def autofeat_engineering(df):
    df = df.copy()
    df = df.drop(columns=["Name", "Ticket", "Cabin", "Embarked"], errors="ignore")
    df = df.select_dtypes(include=[np.number]).dropna()
    if df.shape[0] < 5:
        return df  # Not enough rows to run autofeat
    afr = AutoFeatRegressor(verbose=0, feateng_steps=2)
    try:
        X_feat = afr.fit_transform(df.drop(columns=["Survived"], errors="ignore"), df["Survived"] if "Survived" in df else np.zeros(len(df)))
        return pd.DataFrame(X_feat)
    except:
        return df  # fallback

# --- Featuretools Engineering ---
def featuretools_engineering(df):
    df = df.copy()
    df = df.reset_index(drop=True)
    es = ft.EntitySet(id="titanic_data")
    es = es.add_dataframe(dataframe_name="titanic", dataframe=df, index="index")
    feature_matrix, _ = ft.dfs(entityset=es, target_dataframe_name="titanic", max_depth=2)
    return feature_matrix

# --- Main Playground Function ---
def show_feature_engineering_playground():
    st.header("🧪 Auto Feature Engineering Playground")

    uploaded = st.file_uploader("Upload Titanic dataset (CSV)", type=["csv"], key="autofe")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown("### 🔍 Feature Engineering Strategy")
        option = st.selectbox(
            "Choose a feature engineering method:",
            ["Raw", "Manual (Notebook Inspired)", "Autofeat (Polynomial & Interactions)", "Featuretools (Deep Features)"]
        )

        if option == "Raw":
            st.info("Using the dataset as-is, with no modifications.")
            transformed = df.copy()

        elif option == "Manual (Notebook Inspired)":
            st.info("Applying classic Kaggle-engineered features like Title, IsAlone, FareBin, etc.")
            transformed = manual_engineering(df)

        elif option == "Autofeat (Polynomial & Interactions)":
            st.info("Running autofeat to generate nonlinear polynomial features from numeric columns.")
            transformed = autofeat_engineering(df)

        elif option == "Featuretools (Deep Features)":
            st.info("Using Featuretools to automatically generate deep, stacked features.")
            transformed = featuretools_engineering(df)

        st.markdown("### 🧬 Transformed Dataset Preview")
        st.dataframe(transformed.head())

        csv = transformed.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Transformed CSV", data=csv, file_name="autofe_transformed.csv", mime="text/csv")
