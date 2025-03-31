
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_feature_importance_panel():
    st.title("📊 Feature Importance + Suggestions")

    uploaded = st.file_uploader("Upload a training dataset (with target)", type=["csv"])
    target_col = st.text_input("Enter the target column name")

    if uploaded and target_col:
        df = pd.read_csv(uploaded)
        if target_col not in df.columns:
            st.error(f"❌ '{target_col}' not found in dataset.")
            return

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Use a basic model for importance estimation
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        st.subheader("🔍 Top Feature Importances")
        st.dataframe(importance_df)

        st.subheader("📈 Pareto Chart of Importances")
        fig, ax = plt.subplots()
        ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance (Random Forest)")
        st.pyplot(fig)

        st.subheader("💡 Suggestions")
        top_features = importance_df.head(5)["Feature"].tolist()
        low_features = importance_df.tail(3)["Feature"].tolist()
        st.markdown(f"✅ Consider prioritizing these high-impact features: **{', '.join(top_features)}**")
        st.markdown(f"❌ Consider removing or transforming these lower-impact features: **{', '.join(low_features)}**")
