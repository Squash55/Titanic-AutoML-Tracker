
import streamlit as st

st.set_page_config(page_title="Titanic AutoML Launcher", layout="wide")

st.title("🚢 Titanic AutoML Launcher")
st.markdown("Select your preferred AutoML tool and run your model below.")

automl_tool = st.selectbox("Choose AutoML Tool", ["TPOT", "H2O.ai", "AutoGluon (coming soon)", "PyCaret (coming soon)"])
parallel_mode = st.checkbox("🔁 Run in Parallel (Dask mode)", value=False)
run_button = st.button("🚀 Launch AutoML")

if run_button:
    with st.spinner("Running AutoML..."):
        st.success(f"{automl_tool} run completed! (This is a placeholder result.)")
        st.code("Sample output logs will appear here.", language="bash")

