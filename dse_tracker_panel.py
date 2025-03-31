
import streamlit as st

def show_dse_maturity_panel():
    st.title("📋 DSE Maturity Tracker")

    dse_sections = {
        "1. Create & Collect": ["Define Problem", "Identify Data Sources", "Align with Business"],
        "2. Wrangle & Plan DSEs": ["Clean Data", "Feature Engineering", "EDA Planning"],
        "3. EDA": ["Trends & Patterns", "Correlation", "Outliers"],
        "4. Modeling": ["Baseline Model", "Advanced Models", "HPO"],
        "5. Explain Results": ["Interpretability", "Visualization", "Validation"],
        "6. Deploy & Monitor": ["Production", "Monitoring", "Retraining"]
    }

    # Initialize session_state for DSE tracking if not present
    if "dse_status" not in st.session_state:
        st.session_state.dse_status = {section: {dse: "❌" for dse in dses} for section, dses in dse_sections.items()}

    color_map = {"❌": "gray", "🟡": "orange", "✅": "green"}
    icon_cycle = ["❌", "🟡", "✅"]

    for section, dses in dse_sections.items():
        st.subheader(section)
        cols = st.columns(len(dses))
        for idx, dse in enumerate(dses):
            status = st.session_state.dse_status[section][dse]
            color = color_map[status]
            if cols[idx].button(f"{status} {dse}", key=f"{section}_{dse}"):
                current_index = icon_cycle.index(status)
                next_status = icon_cycle[(current_index + 1) % len(icon_cycle)]
                st.session_state.dse_status[section][dse] = next_status
