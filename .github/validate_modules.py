import os
import ast

TAB_DIR = "tabs"  # or '.' if modules are flat
missing = []
no_run = []

def has_run_function(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        node = ast.parse(f.read())
        return any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in node.body)

for root, _, files in os.walk(TAB_DIR):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            modname = os.path.splitext(file)[0]
            if not has_run_function(path):
                no_run.append(path)

if no_run:
    print("⚠️ Modules missing a run() function:")
    for m in no_run:
        print(" -", m)
else:
    print("✅ All modules have run() defined.")
