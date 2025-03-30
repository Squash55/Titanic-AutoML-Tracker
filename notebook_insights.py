

import streamlit as st

def show_notebook_insights():
    st.header("🧠 Notebook Intelligence Panel")
    st.markdown("Curated insights from top Kaggle Titanic submissions.")


    with st.expander("1. Titanic Top 1% EDA + Feature Engineering + XGBoost (Score: 0.80383)"):
        st.markdown("**Key Features Used:**")
       for feature in note['features']:
        st.markdown(f"- {feature}")
        st.markdown("**Model(s):** " + ", ".join(note['models']))
        st.markdown("[🔗 View Full Notebook](https://www.kaggle.com/code/startupsci/titanic-data-science-solutions)")
    
    with st.expander("2. High Score with Simple Models & Clean Features (Score: 0.79904)"):
        st.markdown("**Key Features Used:**")
        st.markdown("- " + "\n- ".join(note['features']))
        st.markdown("**Model(s):** " + ", ".join(note['models']))
        st.markdown("[🔗 View Full Notebook](https://www.kaggle.com/code/jesucristo/1st-place-solution-a-complete-guide)")
    
    with st.expander("3. Titanic - Top 5% with AutoML + Feature Tuning (Score: 0.79979)"):
        st.markdown("**Key Features Used:**")
        st.markdown("- " + "\n- ".join(note['features']))
        st.markdown("**Model(s):** " + ", ".join(note['models']))
        st.markdown("[🔗 View Full Notebook](https://www.kaggle.com/code/omarelgabry/titanic)")
    
    with st.expander("4. Feature-Rich Logistic Regression Approach (Score: 0.80120)"):
        st.markdown("**Key Features Used:**")
        st.markdown("- " + "\n- ".join(note['features']))
        st.markdown("**Model(s):** " + ", ".join(note['models']))
        st.markdown("[🔗 View Full Notebook](https://www.kaggle.com/code/helgejo/an-interactive-data-science-tutorial)")
    
    with st.expander("5. Neural Net with Feature Engineering (Score: 0.79692)"):
        st.markdown("**Key Features Used:**")
        st.markdown("- " + "\n- ".join(note['features']))
        st.markdown("**Model(s):** " + ", ".join(note['models']))
        st.markdown("[🔗 View Full Notebook](https://www.kaggle.com/code/nadintamer/titanic-survival-predictions-beginner)")
    
