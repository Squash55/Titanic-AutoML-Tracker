# auto_eda.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tpot_connector import __dict__ as _tpot_cache

def interpret_stats_rules(series):
    msg = []
    if series.isnull().sum() > 0:
        msg.append("This column contains missing values.")
    if series.nunique() == 1:
        msg.append("All values are identical — not useful for modeling.")
    if series.dtype in ['int64', 'float64']:
        if series.skew() > 1:
            msg.append("The distribution is right-skewed (long tail on the right).")
        elif series.skew() < -1:
            msg.append("The distribution is left-skewed (long tail on the left).")
        if series.kurt() > 3:
            msg.append("This distribution has heavy tails (high kurtosis).")
        if series.mean() != series.median():
            msg.append(f"The mean ({series.mean():.2f}) and median ({series.median():.2f}) differ — possible outliers.")
    return " ".join(msg) or "No major issues detected statistically."

def interpret_ai_lowtemp(col_name, series):
    # Simulated low-temp AI: formatted summary using real stats
    try:
        summary = f"The feature **{col_name}** has a mean of {series.mean():.2f}, median {series.median():.2f}, and standard deviation {series.std():.2f}."
        if series.skew() > 1:
            summary += " Values are concentrated on the lower end with a few high outliers."
        elif series.skew() < -1:
            summary += " Values are concentrated on the higher end with a few low outliers."
        if series.kurt() > 3:
            summary += " The feature exhibits heavy tails — some extreme values may exist."
        return summary
    except:
        return "AI interpretation is unavailable for non-numeric data."

def run_auto_eda():
    st.subheader("📊 Auto-EDA with Smart Interpretations")

    df = _tpot_cache.get("latest_X_train")

    if df is None:
        st.warning("⚠️ No training data found. Please run AutoML Comparison first.")
        return

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    st.markdown("### 🔍 Numeric Feature Distributions")

    for col in numeric_cols:
        st.markdown(f"#### 📈 {col}")

        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

        st.markdown("**🧠 Rules-Based Interpretation**")
        st.info(interpret_stats_rules(df[col]))

        st.markdown("**🤖 AI-Based Interpretation**")
        st.success(interpret_ai_lowtemp(col, df[col]))

# Auto EDA extension: Parallel Coordinates Plot

import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler

def render_parallel_coordinates(df, stratify_col="Survived", normalize=True):
    st.markdown("### 🧭 Parallel Coordinates Plot")

    filter_col = st.selectbox("Filter by column:", df.columns, index=0)
    unique_vals = df[filter_col].dropna().unique().tolist()
    selected_vals = st.multiselect("Select values to include:", unique_vals, default=unique_vals)

    filtered_df = df[df[filter_col].isin(selected_vals)]

    if normalize:
        norm_df = filtered_df.copy()
        numeric_cols = norm_df.select_dtypes(include=["int64", "float64"]).columns
        scaler = MinMaxScaler()
        norm_df[numeric_cols] = scaler.fit_transform(norm_df[numeric_cols])
        st.caption("🔄 Data normalized using Min-Max scaling.")
    else:
        norm_df = filtered_df.copy()

    plot_cols = st.multiselect("Choose columns to plot:", norm_df.columns, default=norm_df.select_dtypes(include=["float64", "int64"]).columns.tolist()[:5])

    if stratify_col not in norm_df.columns or len(plot_cols) < 2:
        st.warning("Select at least two columns and a valid stratification column.")
        return

    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        parallel_coordinates(norm_df[[stratify_col] + plot_cols], stratify_col, ax=ax, alpha=0.7)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Failed to render parallel coordinates: {e}")