
import streamlit as st
import shap
import xgboost
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -- Load and prepare Titanic data --
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    return df

# -- Load and train sample XGBoost model --
@st.cache_resource
def train_model(df):
    X = df.drop(columns='Survived')
    y = df['Survived']
    model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model, X, y

# -- SHAP interpretation --
def plot_shap_summary(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    st.subheader("📊 SHAP Summary Plot")
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig)

# -- Smart Interpretation --
def smart_explanation():
    st.subheader("🧠 Smart Explanation")
    st.markdown("""
    - **Sex**: The most powerful predictor. Females had much higher survival rates.
    - **Fare**: Higher fare often meant better class and higher survival odds.
    - **Pclass**: First-class passengers had better access to lifeboats and survived more.
    """)

# -- Main app --
# st.set_page_config(page_title="SHAP + Interpretability", layout="wide")
st.title("🔍 SHAP + Interpretability Panel")

df = load_data()
model, X, y = train_model(df)

plot_shap_summary(model, X)
smart_explanation()
