# test_imports.py

import os
import importlib.util
import sys

print("\n🧪 Running Import Test Suite for All .py Files\n")

failures = []

# ✅ Dynamic import of all top-level .py files
for filename in os.listdir("."):
    if filename.endswith(".py") and not filename.startswith("__") and filename != "test_imports.py":
        module_name = filename[:-3]
        print(f"🔍 Testing import of: {module_name}.py", end=" ... ")
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print("✅ PASS")
        except Exception as e:
            print("❌ FAIL")
            failures.append((module_name, str(e)))

# ✅ Manual check of external dependencies (like shap, joblib, etc.)
print("\n📦 Verifying core package imports...\n")

critical_modules = [
    "streamlit", "pandas", "numpy", "sklearn", "tpot", "shap",
    "matplotlib", "joblib", "fpdf", "plotly", "optuna", "seaborn",
    "ptitprince", "xhtml2pdf", "pdfkit", "reportlab"
]

for mod in critical_modules:
    try:
        __import__(mod)
        print(f"✅ {mod}")
    except ImportError as e:
        print(f"❌ {mod} failed: {e}")
        failures.append((mod, str(e)))

# ✅ Summary
if failures:
    print("\n❌ Some modules failed to import or had missing dependencies:")
    for name, err in failures:
        print(f" - {name}: {err}")
    sys.exit(1)
else:
    print("\n✅ All modules and dependencies successfully imported.")
