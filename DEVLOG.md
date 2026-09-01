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
### PostgreSQL Migration + Connection Pooling
**What:** Replaced SQLite with PostgreSQL running in Docker, added SQLAlchemy QueuePool with pool_size=10 and max_overflow=20, added db health check and /ready endpoint
**Why:** SQLite has a single writer lock — under concurrent load from multiple workers every write queues behind the previous one causing latency spikes. PostgreSQL handles thousands of concurrent writes natively and is the standard database for production fintech systems
**Relevance to Track 02:** Razorpay processes 5-7 million transactions per day with spikes to 400-800 per second. An audit log that serialises every write is a liability not an asset. Production credibility requires a database that matches the transaction volume
**Honest note:** psycopg2-binary is used instead of psycopg2 for easier installation — production deployments would compile psycopg2 from source for better performance. The /ready endpoint separates liveness from readiness which is required for Kubernetes but also useful here to ensure Docker does not route traffic before the model is loaded

---
### Multiple Uvicorn Workers
**What:** Changed Dockerfile CMD to run 4 uvicorn worker processes behind a single port, added memory limits and a Docker healthcheck on /ready
**Why:** A single uvicorn process uses one CPU core regardless of how many cores the machine has. With 4 workers, 4 transactions can be scored simultaneously instead of sequentially. Each worker loads its own copy of the model in memory
**Relevance to Track 02:** Razorpay's peak load is 400-800 transactions per second. A single-worker service handling 20-30 per second is a demo, not a product. Four workers pushes this to 80-120 per second on a standard machine, and the architecture scales linearly by adding more workers or more machines
**Honest note:** Each worker loads the 150MB LightGBM model independently into memory so 4 workers use roughly 600MB just for the model. Production would use a model server like Triton that loads the model once and serves all workers from a shared memory space. That is out of scope here but noted for the HLD

---
### Redis Queue for Async Scoring
**What:** Added POST /score/async that enqueues transactions on Redis RQ and returns a job_id immediately, GET /score/result/{job_id} polls for the result, a dedicated rq_worker container processes the queue
**Why:** The synchronous /score endpoint blocks the caller for 50ms while scoring runs. Under burst load — 500 transactions arriving simultaneously — all 500 callers wait in line. With a queue, all 500 are accepted instantly with a job_id. The worker drains the queue at its own pace and the caller polls for results asynchronously. This is how Razorpay's actual risk system would work — the payment gateway cannot block on fraud scoring
**Relevance to Track 02:** Shows production architecture thinking — acceptance and processing are decoupled. The /score endpoint still exists for synchronous use cases like the webhook. The async endpoint handles burst volume without degrading latency for other callers
**Honest note:** RQ with Redis is simpler than Kafka but sufficient for this scale. Kafka would add durability guarantees (messages survive Redis restarts) and exactly-once semantics. For a buildathon demo RQ demonstrates the pattern correctly. Job results are stored in Redis for 300 seconds (result_ttl) before expiring — production would write results to PostgreSQL instead of relying on Redis TTL

---
### Live Fraud Operations Dashboard
**What:** Built a real-time monitoring dashboard at GET /dashboard served directly by FastAPI as an HTML page, auto-refreshing every 10 seconds via /dashboard/stats JSON endpoint showing decision counts, spike alerts, drift status, and the last 10 decisions
**Why:** Fraud operations teams need a live view of what the model is deciding. A spike in BLOCK decisions at 2am is meaningless in a log file but obvious on a dashboard. This is the human-in-the-loop interface for the three-tier decision system
**Relevance to Track 02:** Track 02 asks for bounded and gated money actions. The dashboard is the gate — it surfaces anomalies so a human can intervene. The EWMA spike monitor and PSI drift detector feed directly into the dashboard alerts making monitoring actionable not just measurable
**Honest note:** The dashboard uses plain JavaScript fetch with no framework. In production this would be a proper React app with WebSocket push instead of polling, connected to a time-series database like InfluxDB for historical trends. For a buildathon demo polling every 10 seconds is sufficient and has zero build complexity

---
### High Level Design Document
**What:** Rewrote ARCHITECTURE.md as a full production HLD with ASCII architecture diagram, throughput analysis, bottleneck analysis, privacy design table, future scope, and model card summary
**Why:** A system that cannot be explained to an engineer in one document cannot be trusted in production. The HLD forces honest accounting of what the system can and cannot do at scale
**Relevance to Track 02:** Razorpay evaluates build quality and whether you trust the system. Documenting the four bottlenecks (SHAP latency, model memory, sync DB writes, single RQ worker) with their mitigations shows production thinking that goes beyond the demo
**Honest note:** Current estimated throughput is 60-80 TPS synchronous. Razorpay's peak is 400-800 TPS. The gap is real and documented honestly — closing it requires Triton, Kafka, and horizontal scaling which are in future scope

---
### Data Drift Detection — Live Demonstration
**What:** Created scripts/simulate_drift.py that sends 50 normal transactions matching training distribution then 50 drifted transactions simulating a festival-season coordinated fraud ring (all CASH_OUT, amounts ₹4.5L-₹9.5L vs training mean of ₹500-₹25,000). Fixed drift.py to check amount_log, high_amount_flag, large_amount_flag, and score features. Updated dashboard to render per-feature PSI bar chart with colour-coded alert levels
**Why:** A drift detector that never fires is not a detector — it is a placeholder. Real Indian payment fraud spikes sharply during Diwali and IPL season when transaction volumes and amounts shift dramatically from baseline. The simulation recreates this scenario: a mule network receiving large CASH_OUT transfers from many different senders, which shifts the amount distribution far outside the training range and triggers PSI > 0.2
**Relevance to Track 02:** Track 02 asks for failure handled gracefully. Drift is the silent failure of ML systems — the model continues scoring confidently while the world has changed. Showing PSI fire visibly on the dashboard with a retrain recommendation closes the production ML lifecycle loop: train → deploy → monitor → detect drift → retrain signal
**Honest note:** PSI is computed on the audit log which stores amount and score but not raw features like drain_flag or dest_in_degree. Full feature drift detection would require storing the raw feature vector at inference time — not done here for storage and privacy reasons. The large_amount_flag and amount_log proxy features are sufficient to demonstrate the concept clearly

---
### Dashboard Enhanced — Decision Distribution + PSI Timeline
**What:** Added decision distribution bar showing BLOCK/REVIEW/APPROVE split as a visual percentage bar, added PSI history timeline panel showing drift PSI value at each 10-second check so before/after contrast is visible, expanded last 10 decisions to last 20
**Why:** After running simulate_drift.py the last 10 decisions table showed only BLOCKED transactions because the drifted batch ran last — the normal approved transactions were scrolled off. The distribution bar shows the full 50/50 split clearly. The PSI timeline shows exactly when drift crossed the 0.2 threshold
**Relevance to Track 02:** "Show the audit trail and one failure handled gracefully" — the PSI timeline is the audit trail for model health, not just transaction decisions. A judge can see the exact moment the fraud ring started and when the system detected it
**Honest note:** PSI history is stored in-memory and resets when the service restarts. Production would store drift check results in PostgreSQL with timestamps for long-term model health tracking

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

