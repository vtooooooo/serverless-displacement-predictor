# 🌍 Global Displacement Risk Predictor
### Serverless ML Deployment on AWS — Lab Assignment

> **Topic:** Forced displacement / refugee crisis prediction — the #1 humanitarian crisis of 2026, with 117M+ people displaced globally (UNHCR).

A machine learning system that predicts whether a country is at **HIGH displacement risk** based on humanitarian indicators. Deployed as a fully serverless AWS application: Lambda + API Gateway + S3.

---

## 🏗️ Architecture

```
[S3 Static Dashboard]  →  [API Gateway POST /predict]  →  [Lambda Function]
      (frontend)                  (HTTPS endpoint)           (ML inference)
                                                               model.pkl
                                                               scaler.pkl
```

---

## 📁 Project Structure

```
serverless-displacement-predictor/
├── notebooks/
│   └── 01_train_model.py       # Train & save model artifacts
├── backend/
│   ├── app.py                  # Lambda handler (main entry point)
│   ├── requirements.txt        # scikit-learn, numpy
│   ├── model.pkl               # Trained GBM model (generated)
│   ├── scaler.pkl              # StandardScaler (generated)
│   └── feature_names.pkl       # Feature list (generated)
├── frontend/
│   └── build/
│       └── index.html          # S3 static dashboard
├── data/
│   └── displacement_dataset.csv  # Synthetic training data (generated)
├── package_lambda.py           # Packaging script for Lambda upload
└── README.md
```

---

## 🚀 Deployment Guide

### Prerequisites
- Python 3.10+
- AWS account (Free Tier)
- `pip install scikit-learn numpy pandas`

---

### Step 1 — Train the Model

```bash
python notebooks/01_train_model.py
```

**Output:** `backend/model.pkl`, `backend/scaler.pkl`, `backend/feature_names.pkl`

**Model performance:**
- Algorithm: Gradient Boosting Classifier
- ROC-AUC: **0.93** (5-fold CV)
- Accuracy: **84%** on held-out test set

---

### Step 2 — Test Locally

```bash
python backend/app.py
```

Expected output:
```json
{
  "prediction": 1,
  "label": "High Displacement Risk",
  "risk_tier": "CRITICAL",
  "probability": 0.9987,
  ...
}
```

---

### Step 3 — Package for Lambda

```bash
python package_lambda.py
```

This installs dependencies into `backend/_deps/` and creates `lambda.zip`.

> ⚠️ The zip may be 50–80 MB due to scikit-learn. This is within Lambda's 250 MB limit.

---

### Step 4 — Deploy Lambda Function

1. AWS Console → **Lambda** → **Create function**
2. **Author from scratch**
3. Runtime: **Python 3.12**
4. Upload `lambda.zip` (via S3 if >10 MB — upload zip to S3 first, then provide URL)
5. Handler: `app.handler`
6. Configuration → General:
   - Timeout: **30 seconds**
   - Memory: **256 MB**
7. **Test** with this event:

```json
{
  "body": "{\"conflict_intensity\": 8.5, \"political_instability\": 7.2, \"gdp_per_capita_log\": 6.8, \"food_insecurity_score\": 7.9, \"climate_disaster_risk\": 6.5, \"governance_index\": 2.1, \"human_rights_score\": 8.0, \"unemployment_rate\": 22.0, \"prior_year_displaced\": 250000, \"neighboring_conflict\": 1, \"population_millions\": 42.0}"
}
```

Expected: `statusCode 200` with a JSON prediction.

---

### Step 5 — Create API Gateway Endpoint

1. AWS Console → **API Gateway** → **Create API** → **HTTP API**
2. Integrations → Add → Lambda → select your function
3. Routes → **POST /predict**
4. Deploy → copy the **Invoke URL** (e.g., `https://abc123.execute-api.us-east-1.amazonaws.com`)
5. Test from terminal:

```bash
curl -X POST https://YOUR-API-URL/predict \
  -H "Content-Type: application/json" \
  -d '{"conflict_intensity":8.5,"political_instability":7.2,"gdp_per_capita_log":6.8,"food_insecurity_score":7.9,"climate_disaster_risk":6.5,"governance_index":2.1,"human_rights_score":8.0,"unemployment_rate":22.0,"prior_year_displaced":250000,"neighboring_conflict":1,"population_millions":42.0}'
```

---

### Step 6 — Host Dashboard on S3

1. AWS Console → **S3** → **Create bucket**
   - Bucket name: `displacement-predictor-dashboard` (must be globally unique)
   - Region: same as Lambda
   - **Uncheck** "Block all public access"
2. Upload `frontend/build/index.html`
3. **Properties** → Static website hosting → Enable
   - Index document: `index.html`
4. **Permissions** → Bucket policy → paste:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::YOUR-BUCKET-NAME/*"
  }]
}
```

5. Open the **Bucket website endpoint** URL.

---

### Step 7 — Connect Frontend to API

1. Open the S3 dashboard URL in your browser
2. Paste your **API Gateway Invoke URL** into the endpoint field at the top
3. Load a preset (e.g., 🇸🇩 Sudan) and click **Predict Displacement Risk**
4. Verify the result panel shows a risk tier and probability

---

### Step 8 — Verify End-to-End ✅

| Check | Expected |
|-------|----------|
| S3 dashboard loads | ✅ HTML page visible |
| API returns JSON | ✅ 200 with prediction |
| Sudan preset | ✅ CRITICAL risk |
| Germany preset | ✅ LOW risk |
| No EC2 used | ✅ Fully serverless |

---

## 📊 ML Model Details

### Dataset
- **1,200** synthetic country-year observations
- Based on UNHCR / World Bank humanitarian indicators
- **30% class balance** (high-displacement country-years)

### Features

| Feature | Description | Range |
|---------|-------------|-------|
| `conflict_intensity` | Armed conflict score | 0–10 |
| `political_instability` | Political risk index | 0–10 |
| `gdp_per_capita_log` | Log GDP per capita (USD) | 4–12 |
| `food_insecurity_score` | FAOSTAT food insecurity | 0–10 |
| `climate_disaster_risk` | ND-GAIN disaster risk | 0–10 |
| `governance_index` | World Bank WGI composite | 0–10 |
| `human_rights_score` | Freedom House severity | 0–10 |
| `unemployment_rate` | % labor force | 0–40 |
| `prior_year_displaced` | Lagged displacement count | 0–1M+ |
| `neighboring_conflict` | Neighbor at war (binary) | 0/1 |
| `population_millions` | Country population | 0.1–1400 |

### Top Risk Drivers (by feature importance)
1. **Conflict Intensity** — 23.4%
2. **Food Insecurity** — 14.7%
3. **Political Instability** — 13.9%
4. **Human Rights Score** — 13.7%
5. **Climate Disaster Risk** — 9.7%

---

## 📡 API Reference

**POST /predict**

Request body (JSON):
```json
{
  "conflict_intensity": 8.5,
  "political_instability": 7.2,
  "gdp_per_capita_log": 6.8,
  "food_insecurity_score": 7.9,
  "climate_disaster_risk": 6.5,
  "governance_index": 2.1,
  "human_rights_score": 8.0,
  "unemployment_rate": 22.0,
  "prior_year_displaced": 250000,
  "neighboring_conflict": 1,
  "population_millions": 42.0
}
```

Response:
```json
{
  "prediction": 1,
  "label": "High Displacement Risk",
  "risk_tier": "CRITICAL",
  "probability": 0.9987,
  "confidence_pct": 99.9,
  "top_risk_factors": [
    {"feature": "conflict_intensity", "importance": 0.2343},
    {"feature": "food_insecurity_score", "importance": 0.1474},
    {"feature": "political_instability", "importance": 0.1394}
  ],
  "model_version": "1.0.0"
}
```

**Risk Tiers:**
| Tier | Probability |
|------|-------------|
| LOW | < 30% |
| MODERATE | 30–50% |
| HIGH | 50–75% |
| CRITICAL | ≥ 75% |

---

## 💡 Deliverables Checklist

- [ ] GitHub repository link
- [ ] Public S3 dashboard URL
- [ ] API Gateway endpoint URL
- [ ] Lambda deployment screenshot
- [ ] 5–7 sentence reflection

---

## 🌐 Real-World Connection

This mirrors how humanitarian organizations like UNHCR, ACAPS, and the IRC use ML to triage crisis response resources — predicting where displacement will surge before it happens, enabling pre-positioned aid and diplomatic intervention.

---

*Lab: Serverless AI Deployment on AWS | Data Science Program | University of New Haven*
