import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import shap

try:
    from tpot_connector import latest_tpot_model, latest_X_train
except ImportError:
    latest_tpot_model = None
    latest_X_train = None


# --- Refactored Purpose and Run Function ---
def run():
    show_feature_importance_panel()

def show_feature_importance_panel():
    st.title("📊 Feature Impact & Comparison Analyzer")

    st.markdown("""
    ### 📊 Purpose of the App
    This panel allows you to analyze and compare feature importance between baseline and new datasets. It helps identify which features affect your model's predictions, enabling better decision-making in feature engineering and model optimization.
    """)

    col1, col2 = st.columns(2)
    with col1:
        base_file = st.file_uploader("📁 Upload Baseline CSV", type=["csv"], key="base")
    with col2:
        new_file = st.file_uploader("📁 Upload New Feature CSV", type=["csv"], key="new")

    target_col = st.text_input("🎯 Enter the target column name (must exist in both files)")

    def get_rf_importance(df):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

    def get_shap_importance(model, X):
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        return pd.DataFrame({
            "Feature": X.columns,
            "Importance": mean_abs_shap
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

    if base_file and new_file and target_col:
        df_base = pd.read_csv(base_file)
        df_new = pd.read_csv(new_file)

        if target_col not in df_base.columns or target_col not in df_new.columns:
            st.error("❌ Target column not found in both datasets.")
            return

        method = st.selectbox("📌 Choose Importance Method", ["Random Forest", "SHAP"])

        if method == "Random Forest":
            base_imp = get_rf_importance(df_base)
            new_imp = get_rf_importance(df_new)
        else:
            Xb = df_base.drop(columns=[target_col])
            yb = df_base[target_col]
            Xn = df_new.drop(columns=[target_col])
            yn = df_new[target_col]
            rf_b = RandomForestClassifier().fit(Xb, yb)
            rf_n = RandomForestClassifier().fit(Xn, yn)
            base_imp = get_shap_importance(rf_b, Xb)
            new_imp = get_shap_importance(rf_n, Xn)

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

        # --- Enhanced AI Insights ---
        st.subheader("💡 Smart AI Suggestions")
        gainers = merged[merged["Change"] > 0].head(3)["Feature"].tolist()
        droppers = merged[merged["Change"] < 0].tail(3)["Feature"].tolist()

        if gainers:
            st.markdown(f"✅ These features gained importance: **{', '.join(gainers)}**. Consider optimizing or further exploring these features.")
        if droppers:
            st.markdown(f"⚠️ These features dropped in importance: **{', '.join(droppers)}**. Consider removing, re-engineering, or monitoring these features.")

        st.markdown("""
        ### 🧠 How AI Insights Benefit Feature Importance Comparison
        - **Gainers**: Features that gained importance should be explored further and may offer valuable insights or improvements to the model.
        - **Droppers**: Features that lost importance could be irrelevant, redundant, or noisy. Consider removing them for optimization.
        - **Dynamic Feedback**: AI can help prioritize which features should be further investigated or removed, saving time and resources.
        """)

    elif not base_file and not new_file:
        st.markdown("---")
        st.subheader("📦 No Files? Compare Live TPOT Model")

        model = st.session_state.get("loaded_model", latest_tpot_model)
        X_train = latest_X_train

        if model is not None and X_train is not None:
            st.info("Showing SHAP-based feature importance from current model")
            shap_imp = get_shap_importance(model, X_train)
            st.dataframe(shap_imp.style.background_gradient(axis=0, cmap="Blues"))
            st.bar_chart(shap_imp.set_index("Feature"))
        else:
            st.warning("⚠️ No trained model found. Please run AutoML or load a saved model.")
