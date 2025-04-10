import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tpot_connector import _tpot_cache
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
            "Featuretools (Deep Feature Synthesis)",
            "Logistic Regression Modeling"
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

        elif method == "Logistic Regression Modeling":
            X = st.session_state.get("X") or _tpot_cache.get("latest_X_train")
            y = st.session_state.get("y") or _tpot_cache.get("latest_y_train")

            if X is None or y is None:
                st.warning("⚠️ No dataset found. Please load data or run AutoML first.")
                if st.button("🚀 Launch AutoML Now"):
                    run_automl_launcher()  # Assuming run_automl_launcher is available
                return

            st.markdown("This tool fits logistic regression models with interaction and polynomial terms, and displays model performance and p-values.")

            degree = st.slider("Polynomial Degree", 1, 3, 2)
            include_bias = st.checkbox("Include Bias Term", value=False)

            poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=include_bias)
            X_poly = poly.fit_transform(X)
            feature_names = poly.get_feature_names_out(X.columns)

            st.markdown(f"🧮 Total generated features: `{X_poly.shape[1]}`")

            try:
                logit_model = sm.Logit(y, sm.add_constant(X_poly)).fit(disp=0)
                summary = logit_model.summary2().tables[1]
                summary["Feature"] = ["const"] + list(feature_names)

                # Sort by p-value
                sorted_summary = summary.sort_values("P>|z|")
                top_n = st.slider("Top N terms to display", 5, 30, 10)

                st.subheader("📉 Top Terms by P-Value")
                st.dataframe(sorted_summary[["Feature", "Coef.", "P>|z|"]].head(top_n), use_container_width=True)

                overfit_flags = sorted_summary[sorted_summary["P>|z|"] > 0.05]
                if not overfit_flags.empty:
                    st.warning(f"⚠️ {len(overfit_flags)} terms have p-values > 0.05 — may signal overfitting.")

                # Optional: Fit sklearn version to get AUC
                lr_model = LogisticRegression(max_iter=1000)
                lr_model.fit(X_poly, y)
                y_proba = lr_model.predict_proba(X_poly)[:, 1]
                auc = roc_auc_score(y, y_proba)
                y_pred = lr_model.predict(X_poly)

                st.success(f"✅ Logistic Regression AUC (with poly terms): **{auc:.3f}**")

                st.subheader("📈 Prediction Probability Histogram")
                fig, ax = plt.subplots()
                sns.histplot(y_proba, kde=True, bins=20, ax=ax, color="teal")
                ax.set_title("Predicted Probability Distribution (Train Set)")
                st.pyplot(fig)

                # Visualize confusion matrix
                cm = confusion_matrix(y, y_pred)
                st.subheader("🔍 Confusion Matrix")
                st.text(cm)

                # Save model as .pkl
                if st.button("💾 Export Model (.pkl)"):
                    with open("logreg_poly_model.pkl", "wb") as f:
                        pickle.dump(lr_model, f)
                    st.success("✅ Model exported to logreg_poly_model.pkl")

            except Exception as e:
                st.error(f"❌ LogReg model failed: {type(e).__name__}: {e}")

        st.subheader("🔍 Transformed Features")
        st.dataframe(fe_df.head())
        st.download_button("Download Features CSV", data=fe_df.to_csv(index=False), file_name="engineered_features.csv")
