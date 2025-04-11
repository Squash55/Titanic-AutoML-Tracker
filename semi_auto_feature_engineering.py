import streamlit as st
import pandas as pd

# Feature engineering functions
def add_title(df):
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    return df

def add_family_size(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df

def add_is_alone(df):
    df['IsAlone'] = (df['SibSp'] + df['Parch'] == 0).astype(int)
    return df

def add_fare_bin(df):
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)
    return df

def add_age_group(df):
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 120], labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])
    return df

def add_cabin_known(df):
    df['CabinKnown'] = df['Cabin'].notnull().astype(int)
    return df

# Semi-Automated Feature Engineering tool
def show_semi_automated_feature_engineering():
    st.header("🧪 Semi-Automated Feature Engineering")

    # Purpose statement
    st.markdown("""
    **Purpose**: This tool allows you to **selectively add custom features** to your dataset using **semi-automated** methods. 
    It is a flexible tool that offers suggested transformations like extracting titles from names, creating family size variables, 
    binning fare values, and more, giving you control over the feature engineering process.
    """)

    uploaded = st.file_uploader("Upload Titanic training CSV", type=["csv"], key="feat")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown("**Original Data Preview:**")
        st.dataframe(df.head())

        st.subheader("🔧 Select Features to Add:")
        features = {
            "Title from Name": add_title,
            "FamilySize (SibSp + Parch + 1)": add_family_size,
            "IsAlone": add_is_alone,
            "FareBin (quartiles)": add_fare_bin,
            "AgeGroup (binned)": add_age_group,
            "CabinKnown (missingness flag)": add_cabin_known
        }

        selected = []
        for label in features:
            if st.checkbox(label):
                selected.append(label)

        if selected:
            df_transformed = df.copy()
            for feat in selected:
                df_transformed = features[feat](df_transformed)

            st.subheader("🧬 Transformed Dataset Preview:")
            st.dataframe(df_transformed.head())

            # Allow user to download the transformed dataset
            csv = df_transformed.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Transformed CSV", data=csv, file_name="transformed_titanic.csv", mime="text/csv")

            # AI Insights Section
            st.markdown("### 🧠 AI Insights")
            st.write("""
            **AI Insights** for Feature Engineering:
            - **Feature Selection**: Use AI models to automatically rank features based on predictive power. This can help guide which features are more valuable for training and prediction.
            - **Hyperparameter Tuning**: AI can optimize the parameters for feature transformations, such as the optimal number of bins for the Fare column or the best threshold for age categorization.
            - **Automated Feature Creation**: AI-driven insights can suggest new combinations of features that might help improve the model's performance (e.g., combinations of age, fare, or family size).
            - **Model-Ready Data**: By applying AI to the transformed data, you ensure that the dataset is ready for any predictive modeling task.
            """)

    else:
        st.info("📂 Please upload a Titanic training CSV to begin feature engineering.")

