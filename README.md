# RazorSentry

A cost-aware fraud decisioning service that scores transactions with reason codes, an audit trail, and a rupee-denominated false-positive cost model.

---

## Problem

Online payment fraud costs Indian merchants billions of rupees annually. Existing rule-based systems produce high false-positive rates, blocking legitimate customers and eroding revenue. Static thresholds cannot adapt to shifting fraud patterns, and no existing open tooling quantifies the rupee cost of each misclassification at decision time.

---

## Solution

RazorSentry is a production-structured fraud decisioning service that:

- Scores every transaction in real time using a trained LightGBM model
- Applies a cost-aware threshold calibrated against rupee-denominated false-positive and false-negative costs
- Returns structured reason codes and SHAP-based explanations with every decision
- Maintains a tamper-evident audit log of every decision in SQLite
- Exposes a clean FastAPI interface for synchronous single-transaction scoring, batch scoring, and Razorpay webhook ingestion

---

## Architecture

Transactions arrive at the FastAPI Decision Engine running as 4 parallel uvicorn workers behind a single port. Requests pass through Pydantic input validation and a per-IP rate limiter before entering the scoring pipeline. Two scoring paths exist: POST /score scores synchronously in under 60ms and returns a decision immediately — used for webhooks and low-latency needs. POST /score/async enqueues the transaction on Redis Queue and returns a job_id immediately — used for burst traffic where the caller can tolerate polling. An RQ worker container drains the queue independently. The feature engineering layer extracts 10 signals including velocity, balance error, drain pattern, and graph-flavored in-degree. The LightGBM model (isotonically calibrated) produces a fraud probability. A cost-aware threshold converts that into BLOCK, REVIEW, or APPROVE by maximising rupee net savings. Every decision is written to a PostgreSQL audit log via SQLAlchemy with connection pooling, protected by HMAC-SHA256 PII blinding. A live fraud operations dashboard at GET /dashboard shows real-time decision counts, EWMA spike alerts, PSI drift status, and the last 10 decisions — auto-refreshing every 10 seconds. An optional Groq LLaMA call drafts a 2-line analyst note for REVIEW-queue items only, after the decision is final.

---

## Quickstart

```bash
git clone https://github.com/Sayantan181222/RazorSentry.git
cd RazorSentry
pip install -r requirements.txt

# Download PaySim CSV to data/ (see scripts/download_data.sh)
python src/data_loader.py

python src/train.py

make eval

# Start all services (PostgreSQL, Redis, RazorSentry, RQ worker)
docker compose up --build -d

# Check all containers are healthy
docker compose ps

# Open monitoring dashboard
open http://localhost:8000/dashboard

# Test synchronous scoring
curl -s -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"transaction_id":"TEST_001","step":1,"type":"CASH_OUT","amount":180000,"nameOrig":"C123","oldbalanceOrg":180000,"newbalanceOrig":0,"nameDest":"C456","oldbalanceDest":0,"newbalanceDest":180000}' \
  | python3 -m json.tool

# Test async scoring
curl -s -X POST http://localhost:8000/score/async \
  -H "Content-Type: application/json" \
  -d '{"transaction_id":"TEST_002","step":1,"type":"PAYMENT","amount":1200,"nameOrig":"C789","oldbalanceOrg":50000,"newbalanceOrig":48800,"nameDest":"M111","oldbalanceDest":0,"newbalanceDest":0}' \
  | python3 -m json.tool

# Inspect PostgreSQL audit log
make db-shell

bash scripts/demo_curl.sh
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/score` | Synchronous scoring — returns decision in under 60ms |
| POST | `/score/async` | Async scoring — returns job_id immediately, poll for result |
| GET | `/score/result/{job_id}` | Poll for async scoring result |
| POST | `/batch` | Score a list of transactions sequentially |
| POST | `/webhook/razorpay` | Ingest Razorpay payment.failed webhook events |
| GET | `/decisions/{decision_id}` | Retrieve a past decision by UUID |
| GET | `/decisions` | List recent decisions (default last 50) |
| GET | `/analyst/note/{decision_id}` | Generate Groq LLaMA analyst note for REVIEW decisions |
| GET | `/monitor/spike` | EWMA fraud spike alert |
| GET | `/monitor/drift` | PSI feature drift status |
| GET | `/dashboard` | Live fraud operations dashboard |
| GET | `/dashboard/stats` | Dashboard JSON stats payload |
| GET | `/health` | Service liveness |
| GET | `/ready` | Readiness — 503 until model and DB are both live |

---

## Metrics

Run `make eval` to populate these from the held-out test set.

| Metric | Value |
|---|---|
| PR-AUC | 0.9999 |
| Precision at operating threshold | 1.0000 (100.00%) |
| Recall at operating threshold | 0.9998 (99.98%) |
| Net Savings (INR) | ₹1,953,623,413.78 |
| False Positive Cost Assumption | ₹150/txn |

---

## Evaluation

```bash
make eval
```

This runs `src/eval.py`, which loads the held-out test split, applies the cost-aware threshold sweep, and writes a full classification report, AUC-PR curve, rupee cost summary, sensitivity table, and top-10 false positive cases to `reports/`.

---

## AI Judgment

LightGBM makes every money decision in RazorSentry. The model's output probability is passed through a cost-aware threshold — computed from rupee false-positive and false-negative cost estimates — to produce the final APPROVE / REVIEW / BLOCK verdict. Groq LLaMA (llama-3.1-8b-instant) drafts a 2-line analyst note for REVIEW-queue transactions only. The LLM is never in the decision path. The EWMA spike monitor and PSI drift detector are deliberately LLM-free — pure statistics, fast, auditable.

---

## What Broke and How It Was Fixed

**Temporal leakage in velocity features**
Velocity features were initially computed across the full dataset before the train/test split. PR-AUC appeared inflated at ~0.999. Identified as leakage. Fixed by computing all features strictly within split boundaries. The operating threshold was tuned on a validation split carved from train data only — never on the test set.

**Near-perfect PR-AUC on PaySim synthetic data**
The near-perfect PR-AUC (0.9999) is a known characteristic of the PaySim synthetic dataset — balance-error features alone nearly separate the classes because PaySim encodes fraud with deterministic accounting anomalies. This is documented in the PaySim literature. We report precision and recall at the operating threshold alongside AUC precisely because AUC alone is misleading on synthetic data.

**LabelEncoder at inference time**
LabelEncoder was re-fit on each incoming transaction. A single PAYMENT row encoded as 0 which the model read as CASH_OUT. Fixed by replacing with a hardcoded TYPE_ENCODING dict.

**drain_flag on zero-balance accounts**
amount >= 0.9 * 0.0 is always True so every zero-balance sender got drain_flag=1. Fixed by adding a non-zero guard: oldbalanceOrg > 0 AND amount >= 0.9 * oldbalanceOrg.

**CI/CD: ModuleNotFoundError for src**
pytest could not find the src module in GitHub Actions despite PYTHONPATH and pyproject.toml fixes. Fixed by adding sys.path.insert(0, project_root) directly in conftest.py — works regardless of runner environment configuration.

**Groq model name**
analyst.py was initialized with "groq/compound-mini" which is not a valid Groq model ID. Fixed to "llama-3.1-8b-instant". The except block caught the failure silently so the service never crashed but analyst notes always returned the fallback string.

---

## Model Card

| Field | Detail |
|-------|--------|
| **Training data** | PaySim synthetic dataset — 500,000 legit + all fraud rows (~4,244 fraud in test set) |
| **Fraud rate (train)** | ~0.98% |
| **Fraud rate (test)** | ~4.18% — fraud concentrated in later time steps as expected |
| **Features used** | balance_error_orig, balance_error_dest, drain_flag, zero_orig_after, type_encoded, amount_log, orig_txn_count_1h, orig_txn_sum_1h, dest_in_degree_1h, high_amount_flag |
| **Model** | LightGBM with scale_pos_weight for class imbalance |
| **Calibration** | Isotonic (CalibratedClassifierCV) — threshold moved from 0.02 to 0.35 after calibration |
| **Threshold selection** | Cost-aware rupee net-savings sweep on validation split — not F1, not test set |
| **Operating threshold** | 0.35 (calibrated probabilities — optimal threshold on rupee net savings) |
| **PR-AUC** | 0.9999 (expected on PaySim — synthetic balance-error features nearly perfectly separate classes) |
| **Precision @ threshold** | 1.000 |
| **Recall @ threshold** | 0.9998 |
| **False positive cost** | ₹150 per transaction (review cost assumption) |
| **Net savings (test set)** | ₹1.95 Cr across 101,643 test transactions |

### Known Failure Modes
| Failure Mode | Description |
|-------------|-------------|
| Merchant accounts | PaySim merchant destinations (M...) always have zero balance — balance_error_dest is not meaningful for these |
| Zero-balance senders | drain_flag was fixed to guard against zero-balance false positives (amount >= 90% of 0 is always true) |
| Synthetic data gap | PaySim encodes fraud deterministically — real-world fraud is noisier; PR-AUC will be lower on live data |
| Single-step velocity | Velocity features use step=1 window — may miss slow-burn fraud over many hours |

### Intended Use
- Scoring individual transactions in real time via POST /score
- Batch scoring historical transactions via POST /batch
- Ingesting Razorpay payment webhook events via POST /webhook/razorpay

### Not Intended For
- Replacing human review for high-value transactions above ₹10 lakh
- Deployment on non-PaySim real data without retraining and recalibration
- Offensive fraud (identifying vulnerabilities to exploit) — strictly defense-only

---

## Track

**AI Risk Manager — Track 02, Razorpay Buildathon 2026**

---

## Author

**Sayantan Mandal**
Gati Shakti Vishwavidyalaya
Graduating July 2027
