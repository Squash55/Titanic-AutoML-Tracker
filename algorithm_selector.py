
import streamlit as st

st.set_page_config(page_title="Algorithm Selector", layout="wide")
st.title("🧠 Algorithm Selector (Dual Mode)")
st.markdown("Toggle between Classification and Regression modes to explore algorithm options.")

mode = st.radio("Select Mode:", ["Classification", "Regression"], horizontal=True)

if mode == "Classification":
    algorithms = {
        "Logistic Regression": {
            "Pros": "Interpretable, fast.",
            "Cons": "Assumes linearity, sensitive to outliers.",
            "When to Use": "Simple binary problems, baseline models."
        },
        "Random Forest": {
            "Pros": "High accuracy, robust.",
            "Cons": "Less interpretable.",
            "When to Use": "General-purpose tabular classification."
        },
        "XGBoost": {
            "Pros": "Top performer in competitions.",
            "Cons": "Requires tuning, less interpretable.",
            "When to Use": "Accuracy-critical tabular problems."
        },
        "CatBoost": {
            "Pros": "Handles categorical data natively.",
            "Cons": "Newer, fewer tutorials.",
            "When to Use": "Mixed-type data with categorical features."
        },
        "TabNet": {
            "Pros": "Deep learning for tabular data.",
            "Cons": "Heavier training, less intuitive.",
            "When to Use": "When capturing complex patterns is critical."
        }
    }
else:
    algorithms = {
        "Linear Regression": {
            "Pros": "Simple, interpretable.",
            "Cons": "Assumes linear relationships.",
            "When to Use": "Fast baseline or simple trends."
        },
        "Random Forest Regressor": {
            "Pros": "Accurate, handles non-linearity.",
            "Cons": "Can overfit, less interpretable.",
            "When to Use": "Flexible, general-purpose regression."
        },
        "XGBoost Regressor": {
            "Pros": "Great accuracy, powerful.",
            "Cons": "Requires tuning.",
            "When to Use": "When performance is key."
        },
        "ExtraTrees Regressor": {
            "Pros": "Very fast, robust.",
            "Cons": "Can be noisy.",
            "When to Use": "Fast ensemble alternative to Random Forest."
        },
        "MLP Regressor": {
            "Pros": "Can model complex patterns.",
            "Cons": "Needs tuning and data prep.",
            "When to Use": "Deep relationships in features."
        }
    }

selected_algo = st.selectbox("Choose an Algorithm:", list(algorithms.keys()))
info = algorithms[selected_algo]
st.markdown(f"### ℹ️ {selected_algo} Info")
st.markdown(f"**Pros:** {info['Pros']}")
st.markdown(f"**Cons:** {info['Cons']}")
st.markdown(f"**When to Use:** {info['When to Use']}")
