# smart_poly_finder.py
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def run_smart_poly_finder():
    st.header("🧠 Smart Polynomial Finder")

    if "X" not in st.session_state or "y" not in st.session_state:
        st.warning("❌ No data found in session. Please upload or generate synthetic data first.")
        return

    X = st.session_state.X.copy()
    y = st.session_state.y.copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    feature = st.selectbox("📈 Select numeric feature for nonlinear sweep", numeric_cols)
    max_degree = st.slider("🔁 Max Polynomial Degree to Test", 2, 5, 3)
    p_thresh = st.slider("⚠️ P-Value Threshold for Significance", 0.001, 0.1, 0.05, step=0.005)

    # Store results
    results = []

    for deg in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X_poly = poly.fit_transform(X[[feature]])
        X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out([feature]))

        # Add constant for intercept
        X_with_const = sm.add_constant(X_poly_df)
        model = sm.OLS(y, X_with_const).fit()

        for var, p_val, coef in zip(model.params.index, model.pvalues, model.params):
            if var == 'const':
                continue
            results.append({
                "Feature": feature,
                "Degree": deg,
                "Term": var,
                "P-Value": round(p_val, 5),
                "Coefficient": round(coef, 5),
                "Flag": "✅ Significant" if p_val <= p_thresh else "⚠️ Possibly Overfitting"
            })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # Visualize p-values
    st.markdown("### 📊 P-Value Trend by Degree")
    fig, ax = plt.subplots(figsize=(8, 4))
    for term in results_df["Term"].unique():
        sub = results_df[results_df["Term"] == term]
        ax.plot(sub["Degree"], sub["P-Value"], marker='o', label=term)
    ax.axhline(p_thresh, color='red', linestyle='--', label=f"Threshold = {p_thresh}")
    ax.set_title("P-Values of Polynomial Terms")
    ax.set_xlabel("Degree")
    ax.set_ylabel("P-Value")
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    top_terms = results_df[(results_df["P-Value"] <= p_thresh)].copy()
    if not top_terms.empty:
        add_terms = st.multiselect("➕ Add significant polynomial terms to session", top_terms["Term"].tolist())

        if st.button("📥 Inject Selected Terms"):
            poly = PolynomialFeatures(degree=max_degree, include_bias=False)
            X_poly = poly.fit_transform(X[[feature]])
            X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out([feature]))

            for col in add_terms:
                st.session_state.X[col] = X_poly_df[col]
            st.success(f"✅ Added {len(add_terms)} terms to session features!")
    else:
        st.info("No statistically significant polynomial terms found below your threshold.")
