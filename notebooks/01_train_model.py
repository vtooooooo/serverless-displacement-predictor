"""
=============================================================
  GLOBAL DISPLACEMENT RISK PREDICTOR — MODEL TRAINING
=============================================================
Topic  : Forcible Displacement / Refugee Crisis Prediction
Problem: Predict whether a country will experience HIGH
         displacement outflows in the next year (binary).
Data   : Synthetic dataset based on UNHCR + World Bank
         indicators (conflict score, GDP, governance index,
         food security, disaster risk, etc.)
Output : model.pkl + scaler.pkl + feature_names.pkl
         saved to ../backend/
=============================================================
Run from project root:
    python notebooks/01_train_model.py
=============================================================
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ── Resolve paths relative to THIS script ─────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
BACKEND_DIR = os.path.join(PROJECT_DIR, "backend")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BACKEND_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  1. GENERATE SYNTHETIC DATASET
#     Based on real UNHCR / World Bank structure
# ─────────────────────────────────────────────

N = 1200   # country-year observations

conflict_intensity       = np.random.beta(2, 5, N) * 10
political_instability    = np.random.beta(2, 5, N) * 10
gdp_per_capita_log       = np.random.normal(8.5, 1.5, N)
food_insecurity_score    = np.random.beta(2, 4, N) * 10
climate_disaster_risk    = np.random.beta(1.5, 4, N) * 10
governance_index         = np.random.normal(5, 2, N).clip(0, 10)
human_rights_score       = np.random.beta(2, 3, N) * 10
unemployment_rate        = np.random.beta(2, 5, N) * 40
prior_year_displaced     = np.random.exponential(50000, N)
neighboring_conflict     = np.random.binomial(1, 0.3, N)
population_millions      = np.random.exponential(20, N).clip(0.1, 1400)

logit = (
    0.45 * conflict_intensity
  + 0.30 * political_instability
  - 0.20 * (gdp_per_capita_log - 8.5)
  + 0.25 * food_insecurity_score
  + 0.18 * climate_disaster_risk
  - 0.20 * governance_index
  + 0.22 * human_rights_score
  + 0.01 * unemployment_rate
  + 0.000003 * prior_year_displaced
  + 0.60 * neighboring_conflict
  - 4.5
  + np.random.normal(0, 0.5, N)
)

prob             = 1 / (1 + np.exp(-logit))
high_displacement = (prob > 0.5).astype(int)

print(f"Class balance: {high_displacement.mean():.2%} high-displacement country-years")

df = pd.DataFrame({
    "conflict_intensity":       conflict_intensity,
    "political_instability":    political_instability,
    "gdp_per_capita_log":       gdp_per_capita_log,
    "food_insecurity_score":    food_insecurity_score,
    "climate_disaster_risk":    climate_disaster_risk,
    "governance_index":         governance_index,
    "human_rights_score":       human_rights_score,
    "unemployment_rate":        unemployment_rate,
    "prior_year_displaced":     prior_year_displaced,
    "neighboring_conflict":     neighboring_conflict,
    "population_millions":      population_millions,
    "high_displacement":        high_displacement,
})

csv_path = os.path.join(DATA_DIR, "displacement_dataset.csv")
df.to_csv(csv_path, index=False)
print(f"Dataset saved: {csv_path}  ({len(df)} rows × {df.shape[1]} cols)")

# ─────────────────────────────────────────────
#  2. FEATURE ENGINEERING & SPLIT
# ─────────────────────────────────────────────

FEATURES = [
    "conflict_intensity",
    "political_instability",
    "gdp_per_capita_log",
    "food_insecurity_score",
    "climate_disaster_risk",
    "governance_index",
    "human_rights_score",
    "unemployment_rate",
    "prior_year_displaced",
    "neighboring_conflict",
    "population_millions",
]
TARGET = "high_displacement"

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

# ─────────────────────────────────────────────
#  3. TRAIN MODEL
# ─────────────────────────────────────────────

model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.85,
    random_state=42,
)
model.fit(X_train_sc, y_train)

cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="roc_auc")
print(f"\nCV ROC-AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]

print("\nTest Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# ─────────────────────────────────────────────
#  4. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

importances = pd.Series(model.feature_importances_, index=FEATURES)
print("\nFeature Importances (descending):")
print(importances.sort_values(ascending=False).to_string())

# ─────────────────────────────────────────────
#  5. SAVE ARTIFACTS TO backend/
# ─────────────────────────────────────────────

with open(os.path.join(BACKEND_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(BACKEND_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(BACKEND_DIR, "feature_names.pkl"), "wb") as f:
    pickle.dump(FEATURES, f)

print(f"\n✅ Artifacts saved to {BACKEND_DIR}/")
print("   model.pkl | scaler.pkl | feature_names.pkl")
