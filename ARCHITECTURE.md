# RazorSentry — Architecture

---

## Data Flow

```
PaySim CSV
    │
    ▼
┌─────────────────────┐
│  Feature Engineering │  (src/features.py)
│  - amount ratio      │
│  - balance delta     │
│  - velocity flags    │
│  - tx type encoding  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│     LightGBM        │  (src/train.py / models/*.pkl)
│  fraud probability  │
│  + SHAP values      │
└────────┬────────────┘
         │
         ▼
┌──────────────────────────┐
│  Cost-Aware Threshold    │  (src/eval.py)
│  minimise E[rupee loss]  │
│  → APPROVE/REVIEW/DECLINE│
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  FastAPI Decision Engine │  (src/service.py)
│  POST /score             │
│  POST /batch             │
│  GET  /decisions/{id}    │
│  GET  /health            │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Audit Log (SQLite)      │  (src/audit.py)
│  via SQLAlchemy          │
│  razorsentry.db          │
└──────────────────────────┘
```

---

## Component Descriptions

**Feature Engineering (`src/features.py`)**
Transforms raw PaySim transaction fields into the feature vector expected by the LightGBM model. Key signals include the amount-to-original-balance ratio, the absolute balance delta on both origin and destination accounts, a binary flag for zero-balance drain patterns, and one-hot encoded transaction type. All transformations are deterministic and stateless, making them safe to apply identically at training time and inference time.

**LightGBM Model (`src/train.py`)**
Trains a gradient-boosted decision tree classifier on the engineered feature matrix. Class imbalance is handled via `scale_pos_weight`. MLflow tracks every training run, logging parameters, metrics, and the serialised model artifact. The final model is saved to `models/lgbm_fraud.pkl` for use by the service.

**Cost-Aware Threshold (`src/eval.py`)**
Sweeps the classification threshold across the probability range and computes expected rupee loss at each point using configurable false-positive and false-negative cost weights. The threshold that minimises total expected cost is selected and persisted. This module also produces the AUC-PR curve and full classification report written to `reports/`.

**FastAPI Decision Engine (`src/service.py`)**
Exposes the scoring logic over HTTP. On startup it loads the trained model and threshold from disk. The `/score` endpoint accepts a single transaction JSON, runs feature engineering, calls the model, applies the cost-aware threshold, and returns a structured decision with reason codes and SHAP highlights. The `/batch` endpoint accepts a list of transactions and processes them in one vectorised pass. Every decision is written to the audit log before the response is returned.

**Audit Log (`src/audit.py`)**
Provides a SQLAlchemy-backed persistence layer that records every scoring decision with its transaction ID, timestamp, fraud probability, final verdict, reason codes, and SHAP values serialised as JSON. The `/decisions/{id}` endpoint reads from this store to support post-hoc review and compliance queries.

---

## API Endpoints

| Method | Path               | Description                                      |
|--------|--------------------|--------------------------------------------------|
| POST   | `/score`           | Score a single transaction; returns decision + reason codes |
| POST   | `/batch`           | Score a list of transactions in one call         |
| GET    | `/decisions/{id}`  | Retrieve a past decision by transaction ID       |
| GET    | `/health`          | Service liveness check                           |

---

## Where We Chose NOT to Use an LLM

The decision boundary in RazorSentry is owned entirely by the LightGBM model and the cost-calibrated threshold. An LLM is never consulted during scoring, never influences the APPROVE / REVIEW / DECLINE verdict, and never sees the threshold value. The sole optional use of an LLM is to generate a plain-English analyst note for transactions that have already been assigned to the REVIEW queue by the deterministic model — a purely presentational step that occurs after the decision is final and logged. This design choice ensures the fraud decision is fully reproducible, auditable, and explainable without reference to any probabilistic language model.

---

## Privacy Layer (Software PII Blinding)

Raw account identifiers (nameOrig, nameDest, transaction_id) are never stored
in the audit log. Before any write to SQLite, identifiers are one-way hashed
with HMAC-SHA256 keyed on PII_SALT (an environment secret). The hash is
truncated to 16 hex characters — enough to correlate decisions on the same
account without recovering the original ID.

This is not a hardware TEE (future scope) but provides meaningful protection
against audit log exfiltration. The model scoring pipeline never sees blinded
values — blinding happens after scoring, before storage only.

| What is stored | What is NOT stored |
|---|---|
| HMAC-SHA256[:16] of transaction_id | Raw account number |
| HMAC-SHA256[:16] of nameOrig | Raw sender phone/email |
| Score, decision, reasons, amount | Raw receiver identifier |

Future scope: Hardware TEE (AWS Nitro Enclaves / Intel SGX) for full
cryptographic attestation of the scoring environment.

---

## Decision Policy

Every transaction receives one of three verdicts based purely on the model score. The LLM analyst note is generated after the verdict is final and has no influence on it.

| Score Range | Decision | LLM Involved? |
|---|---|---|
| score >= 0.5 | BLOCK | No |
| score >= operating_threshold | REVIEW + LLM analyst note | Note only, post-decision |
| score < operating_threshold | APPROVE | No |

The `operating_threshold` is determined offline by sweeping thresholds on a validation split and selecting the value that maximises rupee net savings. It is persisted to `models/threshold.txt` and loaded at service startup. It is never recomputed at inference time.
