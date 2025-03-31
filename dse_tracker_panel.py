
import streamlit as st

# Define DSE experiment categories and items
dse_experiments = {
    "A. Data Preparation & Wrangling": [
        "1. Categorical vs. Continuous Variable Comparison",
        "2. Decision Tree Splitting Rules (Univariate)",
        "3. Rule-Based Subgrouping (Multivariate)",
        "4. Label Frequency Patterns",
        "5. Missing Value Handling Methods",
        "6. Outlier Handling Strategies",
        "7. Encoding Strategies",
        "8. Binning of Continuous Variables",
        "9. Dimensionality Reduction Techniques",
        "10. Mathematical Transformations",
        "11. Variable Importance Testing"
    ],
    "B. Modeling Approaches": [
        "12. Base Algorithm Comparisons",
        "13. Optimizer Comparisons",
        "14. Hyperparameter Tuning Strategies",
        "15. Class Imbalance Correction",
        "16. Cross-Validation Strategy Testing",
        "17. Deep Learning vs Classical Models"
    ],
    "C. Model Tuning & Post-Processing": [
        "18. Threshold Tuning (Calibrated Learner)",
        "19. Ensemble Stacking / Blending"
    ],
    "D. Benchmarking & Meta Analysis": [
        "20. Platform Comparisons (AutoML vs Manual)"
    ]
}

# Initialize session state for progress and notes
if "dse_status" not in st.session_state:
    st.session_state.dse_status = {
        section: {dse: "Red" for dse in items} for section, items in dse_experiments.items()
    }

if "dse_notes" not in st.session_state:
    st.session_state.dse_notes = {
        section: {dse: "" for dse in items} for section, items in dse_experiments.items()
    }

def show_dse_maturity_panel():
    st.title("📋 DSE Maturity Tracker")
    st.markdown("Track your progress on the 20 most powerful Data Science Experiments (DSEs).")

    for section, items in dse_experiments.items():
        st.header(section)
        for dse in items:
            col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
            with col1:
                st.markdown(f"**{dse}**")
            with col2:
                if st.button("🔴", key=f"{dse}-red"):
                    st.session_state.dse_status[section][dse] = "Red"
            with col3:
                if st.button("🟡", key=f"{dse}-yellow"):
                    st.session_state.dse_status[section][dse] = "Yellow"
            with col4:
                if st.button("🟢", key=f"{dse}-green"):
                    st.session_state.dse_status[section][dse] = "Green"
            status = st.session_state.dse_status[section][dse]
            st.markdown(f"**Status:** {status}")
            note_key = f"note-{section}-{dse}"
            st.session_state.dse_notes[section][dse] = st.text_input(
                f"Notes for {dse}", 
                value=st.session_state.dse_notes[section][dse],
                key=note_key
            )
