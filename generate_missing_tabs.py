import os
from app import DAIVID_TABS

def create_stub(module_name: str, tab_title: str):
    filename = f"{module_name}.py"
    if os.path.exists(filename):
        return

    content = f'''"""
📊 Auto-generated tab for: {tab_title}
"""

import streamlit as st

@st.cache_data
def run():
    st.title("{tab_title}")
    st.info("✅ This module was auto-generated. Add your content here.")
'''

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Created stub: {filename}")

# Create stubs for all missing modules
for tab_title, module_name in DAIVID_TABS.items():
    create_stub(module_name, tab_title)

print("🏁 Done generating missing tabs.")
