import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import LabelEncoder

def run_shap_screening_doe(df=None, model=None):
    st.title("🧪 SHAP Screening Design of Experiments (DOE)")

    st.markdown("""
    This panel performs a **SHAP-prioritized screening Design of Experiments (DOE)** using the top 8 ranked factors from SHAP values.
    Explore how key drivers affect predictions and uncover their interactions with the model output.
    """)

    if df is None or model is None:
        st.warning("Missing dataset or model. Please ensure both are passed into the DOE panel.")
        return

    df = df.copy()
    if 'Survived' not in df.columns:
        st.error("This DOE panel expects a 'Survived' target column.")
        return

    # Encode categorical variables using LabelEncoder
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=['Survived'])
    y = df['Survived']

    # SHAP ranking of features
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
        top_features = mean_abs_shap.head(8).index.tolist()
    except Exception as e:
        st.error(f"SHAP computation failed: {e}")
        top_features = X.columns[:8].tolist()

    st.markdown("### 🔢 SHAP-Prioritized Features")
    st.info("Factors selected by SHAP as the most influential in the model.")
    st.write(top_features)

    selected_factors = st.multiselect("🎯 Select DOE Factors:", options=X.columns.tolist(), default=top_features)
    if len(selected_factors) < 2:
        st.info("Select at least two factors for interaction exploration.")
        return

    # Main Effects Plot
    st.markdown("### 📈 Main Effects Plot")
    for factor in selected_factors:
        fig, ax = plt.subplots()
        sns.barplot(x=factor, y='Survived', data=df, estimator='mean', ci=None, ax=ax)
        ax.set_title(f"Main Effect: {factor} vs Survival")
        st.pyplot(fig)

    # Interaction Exploration
    st.markdown("### 🔄 Interaction Explorer")
    f1 = st.selectbox("Factor 1:", selected_factors)
    f2 = st.selectbox("Factor 2:", [f for f in selected_factors if f != f1])
    fig, ax = plt.subplots()
    sns.pointplot(x=f1, y='Survived', hue=f2, data=df, ax=ax)
    ax.set_title(f"Interaction: {f1} × {f2} on Survival")
    st.pyplot(fig)

    # DOE Summary Table
    st.markdown("### 📊 Top Factor Combinations")
    summary = df.groupby(selected_factors)['Survived'].mean().reset_index().sort_values(by='Survived', ascending=False)
    st.dataframe(summary.head(20))

    # Auto Interpretation
    st.markdown("### 🧠 Auto Interpretation")
    st.success("Upcoming: This section will highlight statistically interesting interactions, suggest DOE refinements, and visualize marginal effects.")
