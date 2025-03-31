
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title("✅ Matplotlib Test")

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("Test Plot")

st.pyplot(fig)
