import os
import sys
from app import DAIVID_TABS

CHECK_ONLY = "--check-only" in sys.argv

missing_files = []

def create_stub(module_name: str, tab_title: str):
    filename = f"{module_name}.py"
    if os.path.exists(filename):
        return

    if CHECK_ONLY:
        missing_files.append(filename)
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
    print(f"✅ Created: {filename}")

# Loop through all mapped modules
for tab_title, module_name in DAIVID_TABS.items():
    create_stub(module_name, tab_title)

# --- Final Status ---
if CHECK_ONLY:
    if missing_files:
        print("❌ Missing tab files:")
        for f in missing_files:
            print(f" - {f}")
        sys.exit(1)
    else:
        print("✅ All tab files exist.")
else:
    print("🏁 Tab generation complete.")
