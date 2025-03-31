
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def show_feature_importance_panel():
    st.title("📊 Compare Feature Importance (Baseline vs New)")

    col1, col2 = st.columns(2)
    with col1:
        base_file = st.file_uploader("📁 Upload Baseline CSV", type=["csv"], key="base")
    with col2:
        new_file = st.file_uploader("📁 Upload New Feature CSV", type=["csv"], key="new")

    target_col = st.text_input("🎯 Enter the target column name (must exist in both files)")

    if base_file and new_file and target_col:
        df_base = pd.read_csv(base_file)
        df_new = pd.read_csv(new_file)

        if target_col not in df_base.columns or target_col not in df_new.columns:
            st.error("❌ Target column not found in both datasets.")
            return

        def get_importance(df):
            X = df.drop(columns=[target_col])
            y = df[target_col]
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            return pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False).reset_index(drop=True)

        base_imp = get_importance(df_base)
        new_imp = get_importance(df_new)

        st.subheader("📈 Feature Importance Comparison")

        merged = pd.merge(base_imp, new_imp, on="Feature", how="outer", suffixes=("_Base", "_New")).fillna(0)
        merged["Change"] = merged["Importance_New"] - merged["Importance_Base"]
        merged = merged.sort_values("Importance_New", ascending=False)

        st.dataframe(merged.style.background_gradient(axis=0, cmap="RdYlGn", subset=["Change"]))

        st.subheader("📊 Side-by-Side Pareto Charts")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].barh(base_imp["Feature"][::-1], base_imp["Importance"][::-1])
        axes[0].set_title("Baseline Importances")
        axes[1].barh(new_imp["Feature"][::-1], new_imp["Importance"][::-1], color="orange")
        axes[1].set_title("New Importances")
        st.pyplot(fig)

        st.subheader("💡 Smart Suggestions")
        gainers = merged[merged["Change"] > 0].head(3)["Feature"].tolist()
        droppers = merged[merged["Change"] < 0].tail(3)["Feature"].tolist()

        if gainers:
            st.markdown(f"✅ These features gained importance: **{', '.join(gainers)}**")
        if droppers:
            st.markdown(f"⚠️ These features dropped in importance: **{', '.join(droppers)}**")
