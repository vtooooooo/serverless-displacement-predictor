"""
AWS Lambda Handler — Global Displacement Risk Predictor
--------------------------------------------------------
Receives a JSON POST body with country risk indicators,
returns a displacement risk prediction + probability.

Expected input (all fields required):
{
  "conflict_intensity":     float  0-10
  "political_instability":  float  0-10
  "gdp_per_capita_log":     float  (log of GDP per capita in USD, e.g. 7.6 = ~$2000)
  "food_insecurity_score":  float  0-10
  "climate_disaster_risk":  float  0-10
  "governance_index":       float  0-10
  "human_rights_score":     float  0-10
  "unemployment_rate":      float  0-40  (percent)
  "prior_year_displaced":   float  (number of people displaced last year)
  "neighboring_conflict":   int    0 or 1
  "population_millions":    float
}
"""

import json
import pickle
import numpy as np
import os

# ── Load model artifacts once (Lambda container reuse) ──────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE, "feature_names.pkl"), "rb") as f:
    FEATURES = pickle.load(f)

# ── Risk tier labels ─────────────────────────────────────────────────────────
def risk_tier(prob):
    if prob >= 0.75:
        return "CRITICAL"
    elif prob >= 0.50:
        return "HIGH"
    elif prob >= 0.30:
        return "MODERATE"
    else:
        return "LOW"


def handler(event, context):
    """Main Lambda entry point."""

    # ── CORS preflight ────────────────────────────────────────────────────────
    if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": cors_headers(),
            "body": "",
        }

    try:
        # ── Parse body ────────────────────────────────────────────────────────
        body = event.get("body", "{}")
        if isinstance(body, str):
            body = json.loads(body)

        # ── Validate & extract features ───────────────────────────────────────
        missing = [f for f in FEATURES if f not in body]
        if missing:
            return error_response(400, f"Missing fields: {missing}")

        x = np.array([[float(body[f]) for f in FEATURES]])
        x_scaled = scaler.transform(x)

        # ── Predict ───────────────────────────────────────────────────────────
        prediction = int(model.predict(x_scaled)[0])
        probability = float(model.predict_proba(x_scaled)[0][1])

        # ── Feature contributions (top 3) ─────────────────────────────────────
        importances = dict(zip(FEATURES, model.feature_importances_))
        top_factors = sorted(importances.items(), key=lambda kv: -kv[1])[:3]

        result = {
            "prediction":        prediction,
            "label":             "High Displacement Risk" if prediction == 1 else "Low Displacement Risk",
            "risk_tier":         risk_tier(probability),
            "probability":       round(probability, 4),
            "confidence_pct":    round(probability * 100, 1) if prediction == 1 else round((1 - probability) * 100, 1),
            "top_risk_factors":  [{"feature": k, "importance": round(v, 4)} for k, v in top_factors],
            "model_version":     "1.0.0",
        }

        return {
            "statusCode": 200,
            "headers": cors_headers(),
            "body": json.dumps(result),
        }

    except json.JSONDecodeError:
        return error_response(400, "Invalid JSON body")
    except ValueError as e:
        return error_response(400, f"Value error: {str(e)}")
    except Exception as e:
        return error_response(500, f"Internal error: {str(e)}")


def cors_headers():
    return {
        "Content-Type":                "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


def error_response(status, message):
    return {
        "statusCode": status,
        "headers": cors_headers(),
        "body": json.dumps({"error": message}),
    }


# ── Local testing ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_event = {
        "body": json.dumps({
            "conflict_intensity":    8.5,
            "political_instability": 7.2,
            "gdp_per_capita_log":    6.8,
            "food_insecurity_score": 7.9,
            "climate_disaster_risk": 6.5,
            "governance_index":      2.1,
            "human_rights_score":    8.0,
            "unemployment_rate":     22.0,
            "prior_year_displaced":  250000,
            "neighboring_conflict":  1,
            "population_millions":   42.0,
        })
    }
    resp = handler(test_event, None)
    print(json.dumps(json.loads(resp["body"]), indent=2))
