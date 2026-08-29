# RazorSentry — Development Log
Author: Sayantan Mandal, Gati Shakti Vishwavidyalaya
Track: AI Risk Manager — Track 02, Razorpay Buildathon

> Entries are feature-by-feature.
> Each entry explains what was built, why, and the honest tradeoff.

---
### Time-Ordered Train/Test Split
**What:** Implemented in `src/data_loader.py` to subsample 500k legit rows and all fraud rows, sorting by step and splitting 80/20.
**Why:** Prevents temporal data leakage — future transactions must not inform past decisions.
**Relevance to Track 02:** Honest metrics require leakage-free splits; PR-AUC of 0.9999 on PaySim is real not leaked.
**Honest note:** PaySim fraud concentrates in later steps so test fraud rate (4.18%) is higher than train (0.98%) — expected for synthetic data.

---
### Feature Engineering (10 features)
**What:** Implemented in `src/features.py` computing accounting errors, drain flag, velocity counts/sums, and receiver in-degrees.
**Why:** Balance-error features catch accounting anomalies that rules alone miss.
**Relevance to Track 02:** "Problem taste" — picking features that actually reflect payment fraud mechanics.
**Honest note:** LabelEncoder was initially re-fit at inference time giving wrong type encodings — fixed by hardcoding TYPE_ENCODING dict.

---
### Rules Baseline + LightGBM with Cost-Aware Threshold
**What:** Implemented in `src/train.py` comparing a heuristic rule baseline against LightGBM optimized for net rupee savings.
**Why:** Rules alone catch 49.7% of fraud at 12.4% precision; the model catches 100% at 90.5% precision.
**Relevance to Track 02:** Track 02 bar requires measured precision and recall on a held-out test set.
**Honest note:** Isotonic calibration (CalibratedClassifierCV) was applied post-training, moving operating threshold from 0.02 to an interpretable 0.35.

---
### Append-Only Audit Log
**What:** Implemented in `src/audit.py` using SQLite and SQLAlchemy to log every scored transaction, score, reasons, and latency.
**Why:** Every fraud decision must be explainable and retrievable — regulators require immutable records.
**Relevance to Track 02:** "Every money action explainable, bounded and gated" from the buildathon bar.
**Honest note:** SQLite is sufficient for this scale; production would use a write-once ledger like DynamoDB with point-in-time recovery.

---
### FastAPI Decision Service with SHAP Reason Codes
**What:** Implemented in `src/service.py` exposing `POST /score`, `POST /batch`, `GET /decisions/{id}`, and `GET /health`.
**Why:** A model score alone is not actionable — analysts need to know WHY a transaction was flagged.
**Relevance to Track 02:** "Show the audit trail and one failure handled gracefully" — SHAP maps features to plain English reasons.
**Honest note:** SHAP TreeExplainer adds ~10-15ms latency per transaction — acceptable for fraud review, not for sub-millisecond payment authorisation.

---
### Three-Tier Decision Policy (BLOCK / REVIEW / APPROVE)
**What:** Implemented in `src/service.py` via `_apply_policy()` using operating threshold and block threshold.
**Why:** Hard blocking every flagged transaction loses legitimate revenue — the REVIEW tier lets humans decide on borderline cases.
**Relevance to Track 02:** False positive cost is a first-class metric in Track 02; the three tiers make FP cost explicit.
**Honest note:** drain_flag initially fired on zero-balance accounts (amount >= 0.9 * 0 is always true) — fixed with a non-zero guard.

---
### Groq LLaMA Analyst Notes for REVIEW Queue
**What:** Implemented in `src/analyst.py` exposing `GET /analyst/note/{decision_id}` powered by `llama-3.1-8b-instant`.
**Why:** A 2-line plain-English note reduces analyst cognitive load when reviewing borderline transactions.
**Relevance to Track 02:** "AI judgment — the right tool in the right place" — LLM drafts notes only after the model has already decided; never in the decision path.
**Honest note:** If GROQ_API_KEY is missing the function returns a safe fallback string — the service never crashes on LLM failure.

---
### EWMA Fraud Spike Monitor
**What:** Implemented in `src/monitor.py` exposing `GET /monitor/spike` using exponential weighted moving average of flagged rates.
**Why:** A single bad transaction is noise; a spike in flagged rate signals a coordinated attack or data quality issue.
**Relevance to Track 02:** "Failure handled gracefully" — the monitor catches systemic problems before they compound.
**Honest note:** Deliberately LLM-free — EWMA is fast, auditable, and has no hallucination risk; LLM spike detection would add latency and unpredictability.

---
### Razorpay Webhook Endpoint
**What:** Implemented in `src/service.py` exposing `POST /webhook/razorpay` to ingest `payment.failed` event payloads.
**Why:** RazorSentry must accept real Razorpay payment event shapes to be usable in the buildathon context.
**Relevance to Track 02:** Directly ties the project to Razorpay's payment infrastructure; routes payment.failed events through the same scoring pipeline.
**Honest note:** Balance fields (oldbalanceOrg, newbalanceOrig) are set to 0.0 for webhook inputs since Razorpay does not expose them — balance-error features will not fire for webhook-sourced transactions.

---
### Software PII Blinding
**What:** Designed for `src/privacy.py` using salted HMAC-SHA256 hashing on origin and destination identifiers.
**Why:** Financial audit logs must never store raw account identifiers in plaintext — DPDP Act 2023 compliance.
**Relevance to Track 02:** Track 02 judges financial data systems; showing privacy-by-design separates this from student projects.
**Honest note:** HMAC-SHA256[:16] is one-way and cannot be reversed — this means audit records cannot be linked back to the original account without the PII_SALT secret, which is intentional. Verification confirmed: API response returns original ID to caller, DB stores only the 16-char HMAC hash — correct behaviour by design.

---
### PSI Feature Drift Detector
**What:** Designed for `src/drift.py` exposing `GET /monitor/drift` calculating Population Stability Index against baseline distributions.
**Why:** A model trained on PaySim synthetic data will degrade when real transaction distributions shift — PSI catches this before precision drops.
**Relevance to Track 02:** Production ML systems need monitoring beyond accuracy; showing PSI demonstrates understanding of model lifecycle.
**Honest note:** PSI checks only amount_log and high_amount_flag via the audit log — full feature drift requires raw transaction replay which is not stored for privacy reasons.

---
### Input Validation and Rate Limiting
**What:** Configured in `src/service.py` via Pydantic field constraints, amount checks, and middleware.
**Why:** An unvalidated API in a financial system is a liability — invalid types and negative amounts must be rejected before scoring.
**Relevance to Track 02:** "Build quality — does it run, is it structured, would you trust it".
**Honest note:** Rate limiting is per-IP which is bypassable behind a shared NAT — production would use API keys with per-merchant limits.

---
### Full Pipeline Rerun + PII Blinding Verification
**What:** Reran train.py and eval.py after calibration and PII blinding were added, verified blinding works correctly end to end
**Why:** Calibration changes the model internals so all downstream metrics needed to be regenerated from scratch
**Relevance to Track 02:** Honest metrics — all numbers in the repo reflect the current model not a stale run
**Honest note:** top_fp_cases.csv was silently empty due to a hardcoded threshold — caught and fixed during this rerun

---
## What Broke and How It Was Fixed

**Temporal leakage in velocity features**
Velocity features were initially computed across the full dataset before the train/test split.
PR-AUC appeared inflated at ~0.999. Identified as leakage. Fixed by computing all features
strictly within split boundaries.

**LabelEncoder at inference time**
LabelEncoder was re-fit on each incoming transaction. A single PAYMENT row encoded as 0
which the model read as CASH_OUT. Fixed by replacing with a hardcoded TYPE_ENCODING dict.

**drain_flag on zero-balance accounts**
amount >= 0.9 * 0.0 is always True so every zero-balance sender got drain_flag=1.
Fixed by adding a non-zero guard: oldbalanceOrg > 0 AND amount >= 0.9 * oldbalanceOrg.

**CI/CD: ModuleNotFoundError for src**
pytest could not find the src module in GitHub Actions despite PYTHONPATH and pyproject.toml fixes.
Fixed by adding sys.path.insert(0, project_root) directly in conftest.py — works regardless
of runner environment configuration.
Also by adding the __init__.py file to the tests directory.

**Groq model name**
analyst.py was initialized with "groq/compound-mini" which is not a valid Groq model ID.
Fixed to "llama-3.1-8b-instant". The except block caught the failure silently so the service
never crashed but analyst notes always returned the fallback string.

**top_fp_cases.csv was empty despite false positives existing**
save_top_fp_cases used a hardcoded threshold of 0.5 to build the false positive mask.
The operating threshold is much lower so no legitimate transaction scored above 0.5
on well-separated PaySim data — the mask returned zero rows and the CSV had only headers.
Fixed by passing operating_threshold as a parameter so the mask uses the actual decision boundary.

**Audit log contained 2006 unblinded records from pre-privacy runs**
Records written before src/privacy.py was integrated stored raw transaction IDs
in plaintext. The blinding code was correctly wired in service.py at line 185
but old records persisted. Confirmed blinding works on new records:
SENSITIVE_ACCOUNT_99999 stored as 2e10c17cfbb44f3d in the DB.
Fixed by deleting all records where length(transaction_id) != 16.

---

## Production Hardening
- PII blinding: HMAC-SHA256 hashes account IDs before audit log storage
- Isotonic calibration: probability scores now interpretable, threshold corrected
- PSI drift detector: monitors feature distribution shift from training baseline
- Input validation: Pydantic validators reject invalid types, negative amounts, bad steps
- Rate limiting: 60 req/min on /score, 10 req/min on /batch via slowapi
- All changes tested and pushed

