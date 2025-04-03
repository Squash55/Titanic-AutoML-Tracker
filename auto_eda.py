import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from tpot_connector import __dict__ as _tpot_cache
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

def run_auto_eda():
    st.title("📊 Auto EDA Dashboard")

    df = _tpot_cache.get("latest_X_train")
    y = _tpot_cache.get("latest_y_train")
    model = _tpot_cache.get("latest_tpot_model") or _tpot_cache.get("latest_rf_model")

    if df is None or y is None:
        st.warning("Train a model first to populate Auto EDA.")
        return

    df = df.copy()
    df['target'] = y

    tab = st.selectbox("Select EDA View", [
        "Main Effects",
        "Pairwise Scatter + R²",
        "Parallel Coordinates",
        "Nomogram",
        "Raincloud",
        "Jittered Categorical Plot",
        "3D Rotating Surface"
    ])

    if tab == "Main Effects":
        st.subheader("📈 Main Effects Plot")
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 6))
        for i, col in enumerate(df.select_dtypes("number").columns[:8]):
            row, coln = divmod(i, 4)
            sns.lineplot(x=df[col], y=df['target'], ax=axes[row, coln])
            axes[row, coln].set_title(col)
        st.pyplot(fig)

    elif tab == "Pairwise Scatter + R²":
        st.subheader("📉 Linear Scatter + R²")
        feat_x = st.selectbox("X Variable", df.columns)
        feat_y = "target"
        model = LinearRegression()
        X_ = df[[feat_x]].values
        y_ = df[feat_y].values
        model.fit(X_, y_)
        r2 = model.score(X_, y_)
        fig = px.scatter(df, x=feat_x, y=feat_y, trendline="ols", title=f"R² = {r2:.2f}")
        st.plotly_chart(fig)

    elif tab == "Parallel Coordinates":
        st.subheader("🪄 Parallel Coordinates (Normalized)")
        num_df = df.select_dtypes(include=["number"]).copy()
        if len(num_df.columns) < 2:
            st.warning("Not enough numeric columns.")
            return
        scaler = MinMaxScaler()
        norm_df = pd.DataFrame(scaler.fit_transform(num_df), columns=num_df.columns)
        fig = px.parallel_coordinates(norm_df, color="target")
        st.plotly_chart(fig)

    elif tab == "Nomogram":
        st.subheader("🧮 Nomogram Simulation")
        model_type = st.radio("Select model", ["TPOT", "RandomForest"], horizontal=True)
        model = _tpot_cache.get("latest_tpot_model") if model_type == "TPOT" else _tpot_cache.get("latest_rf_model")
        if model is None:
            st.warning("Model not available.")
            return

        try:
            explainer = shap.Explainer(model, df.drop("target", axis=1))
            shap_vals = explainer(df.drop("target", axis=1))
            shap_imp = np.abs(shap_vals.values).mean(axis=0)
            ranked = dict(sorted(zip(df.columns[:-1], shap_imp), key=lambda x: x[1], reverse=True))
            top_features = list(ranked.keys())[:8]
        except:
            top_features = df.columns[:8]

        input_vals = {}
        for col in top_features:
            input_vals[col] = st.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        input_df = pd.DataFrame([input_vals])
        try:
            prob = model.predict_proba(input_df)[0][1]
            st.success(f"Prediction = {prob:.3f}")
            shap_val = explainer(input_df)
            fig = shap.plots.waterfall(shap_val[0], show=False)
            st.pyplot(fig)
        except:
            st.warning("SHAP or prediction failed.")

    elif tab == "Raincloud":
        st.subheader("🌧️ Raincloud Plot")
        import ptitprince as pt
        fig, ax = plt.subplots(figsize=(10, 5))
        pt.RainCloud(x='target', y=df.columns[0], data=df, palette="Set2", bw=.2, width_viol=.6, ax=ax)
        st.pyplot(fig)

    elif tab == "Jittered Categorical Plot":
        st.subheader("🔴🔵 Jittered Category Scatter")
        cat_col = st.selectbox("Choose category", df.select_dtypes("object").columns.tolist() or ["None"])
        if cat_col == "None":
            st.warning("No categorical data found.")
            return
        fig = px.strip(df, x=cat_col, y="target", color="target", stripmode="overlay", jitter=0.3)
        st.plotly_chart(fig)

    elif tab == "3D Rotating Surface":
        st.subheader("🌀 3D Rotating Readiness Surface")
        x = st.selectbox("X axis", df.select_dtypes("number").columns)
        y = st.selectbox("Y axis", df.select_dtypes("number").columns, index=1)
        z = "target"
        surface_model = LinearRegression()
        X_grid = df[[x, y]].values
        Z_target = df[z].values
        surface_model.fit(X_grid, Z_target)
        grid_x, grid_y = np.meshgrid(
            np.linspace(df[x].min(), df[x].max(), 30),
            np.linspace(df[y].min(), df[y].max(), 30)
        )
        grid_pred = surface_model.predict(np.c_[grid_x.ravel(), grid_y.ravel()]).reshape(grid_x.shape)
        fig = go.Figure(data=[
            go.Surface(z=grid_pred, x=grid_x, y=grid_y, opacity=0.6),
            go.Scatter3d(x=df[x], y=df[y], z=df[z], mode='markers', marker=dict(size=3, color=df[z], colorscale='Viridis'))
        ])
        st.plotly_chart(fig)
