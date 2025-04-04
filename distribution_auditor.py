# distribution_auditor.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from tpot_connector import _tpot_cache

def best_fit_distribution(data):
    DISTRIBUTIONS = [
        stats.norm, stats.expon, stats.gamma, stats.beta, stats.lognorm,
        stats.weibull_min, stats.weibull_max, stats.uniform, stats.t, stats.triang
    ]
    
    best_fit_name = None
    best_p = -1
    best_stat = None

    for distribution in DISTRIBUTIONS:
        try:
            params = distribution.fit(data)
            D, p = stats.kstest(data, distribution.name, args=params)
            if p > best_p:
                best_p = p
                best_fit_name = distribution.name
                best_stat = D
        except Exception:
            continue
    return best_fit_name, best_p, best_stat

def run_distribution_auditor():
    st.title("📈 Feature Distribution Auditor + KS Test")
    st.markdown("This module tests each numeric feature against multiple known distributions to find the best fit.")

    df = _tpot_cache.get("X_train")
    if df is None:
        st.warning("⚠️ No training data found. Please run AutoML first.")
        return

    st.markdown("### 🔍 Analyzing Feature Distributions...")
    numeric_cols = df.select_dtypes(include='number').columns
    distribution_summary = []

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        best_fit, p_val, ks = best_fit_distribution(series)
        distribution_summary.append({
            "Feature": col,
            "Best Fit Distribution": best_fit,
            "KS p-value": round(p_val, 4),
            "KS Statistic": round(ks, 4)
        })

    results_df = pd.DataFrame(distribution_summary)
    st.dataframe(results_df)

    st.markdown("---")
    st.markdown("### 🧠 How to Interpret the KS Test")
    st.info("""
The **Kolmogorov–Smirnov (KS) test** compares two distributions to see if they are statistically different.

- **KS p ≥ 0.05** → ✅ Distributions are likely similar (*fail to reject* null hypothesis).
- **KS p < 0.05** → ⚠️ Distributions are likely different (*reject* null hypothesis).

DAIVID uses KS testing to:
- Detect feature drift between train/test sets
- Match features to known statistical distributions
- Suggest power transformations or normalization
    """)

    st.markdown("✅ Best-fit distributions can guide preprocessing decisions like normalization, binning, or log transforms.")
