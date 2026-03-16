#!/usr/bin/env python3
"""
package_lambda.py
-----------------
Installs dependencies into backend/ and creates lambda.zip
ready for upload to AWS Lambda.

Usage (run from project root):
    python package_lambda.py
"""

import os
import subprocess
import zipfile
import shutil
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(PROJECT_DIR, "backend")
ZIP_OUT     = os.path.join(PROJECT_DIR, "lambda.zip")
DEPS_DIR    = os.path.join(BACKEND_DIR, "_deps")

print("=" * 55)
print("  Lambda Packaging Script")
print("  Global Displacement Risk Predictor")
print("=" * 55)

# ── Step 1: Install dependencies into _deps/ ──────────────────
print("\n[1/3] Installing Python dependencies into backend/_deps/ ...")
reqs = os.path.join(BACKEND_DIR, "requirements.txt")
if not os.path.exists(reqs):
    print("  ERROR: backend/requirements.txt not found.")
    sys.exit(1)

subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "-r", reqs,
     "-t", DEPS_DIR,
     "--quiet"],
    check=True
)
print("  ✅ Dependencies installed.")

# ── Step 2: Check required model artifacts ────────────────────
print("\n[2/3] Checking model artifacts ...")
required = ["model.pkl", "scaler.pkl", "feature_names.pkl", "app.py"]
missing  = [f for f in required if not os.path.exists(os.path.join(BACKEND_DIR, f))]
if missing:
    print(f"  ERROR: Missing files: {missing}")
    print("  Run:  python notebooks/01_train_model.py  first.")
    sys.exit(1)
print("  ✅ All artifacts present.")

# ── Step 3: Create lambda.zip ─────────────────────────────────
print(f"\n[3/3] Building {ZIP_OUT} ...")
if os.path.exists(ZIP_OUT):
    os.remove(ZIP_OUT)

with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    # Add dependencies
    for root, dirs, files in os.walk(DEPS_DIR):
        for file in files:
            abs_path = os.path.join(root, file)
            arc_name = os.path.relpath(abs_path, DEPS_DIR)
            zf.write(abs_path, arc_name)
    # Add backend files (not _deps itself)
    for fname in ["app.py", "model.pkl", "scaler.pkl", "feature_names.pkl"]:
        zf.write(os.path.join(BACKEND_DIR, fname), fname)

size_mb = os.path.getsize(ZIP_OUT) / (1024 * 1024)
print(f"  ✅ lambda.zip created  ({size_mb:.1f} MB)")

print("\n" + "=" * 55)
print("  NEXT STEPS")
print("=" * 55)
print("  1. Go to AWS Lambda → Create Function")
print("     Runtime: Python 3.12")
print("     Handler: app.handler")
print("  2. Upload lambda.zip")
print("  3. Set timeout: 30s, Memory: 256MB")
print("  4. Test with the sample event in README.md")
print("=" * 55)
