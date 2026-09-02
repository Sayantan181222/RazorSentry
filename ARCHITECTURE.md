# RazorSentry — High Level Design

Author: Sayantan Mandal, Gati Shakti Vishwavidyalaya
Track: AI Risk Manager — Track 02, Razorpay Buildathon

---

## System Overview

RazorSentry is a cost-aware fraud decisioning service built for Indian payment
infrastructure. It scores transactions in real time, explains every decision with
SHAP reason codes, and writes an immutable PII-blinded audit log to PostgreSQL.

Two scoring paths handle different load profiles:
- Synchronous path (POST /score): sub-60ms, used for webhooks and real-time decisions
- Async path (POST /score/async): accepts instantly, queued via Redis RQ, polled by caller

---

## Production Architecture Diagram
```
                     ┌─────────────────────────────────────────────┐
                     │           INCOMING TRAFFIC                  │
                     │   Razorpay webhooks / merchant API calls    │
                     └─────────────────┬───────────────────────────┘
                                       │
                                       ▼
                     ┌─────────────────────────────────────────────┐
                     │         LOAD BALANCER (future: nginx)       │
                     │         Currently: Docker port 8000         │
                     └─────────────────┬───────────────────────────┘
                                       │
                      ┌────────────────┼────────────────┐
                      │                │                │
                      ▼                ▼                ▼
           ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
           │  uvicorn     │  │  uvicorn     │  │  uvicorn     │
           │  worker 1    │  │  worker 2    │  │  worker 3/4  │
           │  (FastAPI)   │  │  (FastAPI)   │  │  (FastAPI)   │
           └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                  │                 │                  │
      ┌───────────┴─────────────────┴──────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        REQUEST ROUTING                          │
│                                                                 │
│   POST /score          ──►  SYNC SCORING PATH                   │
│   POST /webhook/razorpay                                        │
│                                                                 │
│   POST /score/async    ──►  ASYNC SCORING PATH                  │
└──────────┬─────────────────────────┬────────────────────────────┘
           │                         │
           ▼                         ▼
┌──────────────────┐      ┌─────────────────────────┐
│    SYNC PATH     │      │       ASYNC PATH        │
│                  │      │                         │
│  1. Validate     │      │  1. Validate            │
│  2. Build feats  │      │  2. Enqueue to Redis RQ │
│  3. LGB score    │      │  3. Return job_id       │
│  4. SHAP reasons │      │  4. Caller polls result │
│  5. Log to PG    │      │                         │
│  6. Return       │      │  RQ WORKER (separate    │
│                  │      │  container):            │
│  Latency: <60ms  │      │  - Dequeues job         │
└─────────┬────────┘      │  - Scores transaction   │
          │               │  - Logs to PostgreSQL   │
          │               └─────────────┬───────────┘
          │                             │
          └──────────────┬──────────────┘
                         │
                         ▼
           ┌─────────────────────────┐
           │   PII BLINDING LAYER    │
           │   HMAC-SHA256[:16]      │
           │   transaction_id        │
           │   nameOrig, nameDest    │
           └─────────────┬───────────┘
                         │
                         ▼
           ┌─────────────────────────┐
           │  POSTGRESQL AUDIT LOG   │
           │  decisions table        │
           │  pool_size=10           │
           │  max_overflow=20        │
           │  pool_pre_ping=True     │
           └─────────────┬───────────┘
                         │
       ┌─────────────────┴────────────────────────────┐
       │                                              │
       ▼                                              ▼
┌──────────────────────┐              ┌────────────────────────┐
│       MONITORS       │              │       DASHBOARD        │
│                      │              │   GET /dashboard       │
│ EWMA Spike Monitor   │              │   Auto-refresh 10s     │
│ GET /monitor/spike   │              │   Decision counts      │
│                      │              │   Spike / drift alerts │
│ PSI Drift Monitor    │              │   Last 10 decisions    │
│ GET /monitor/drift   │              └────────────────────────┘
└──────────────────────┘
```

---

## Component Reference

**FastAPI Workers (src/service.py)**
Four uvicorn worker processes share port 8000. Each loads its own model copy at
startup. Handles input validation, rate limiting (60/min sync, 200/min async),
scoring, SHAP explanation, PII blinding, and audit log writes.

**LightGBM Model (src/train.py)**
Gradient-boosted classifier trained on 10 payment-domain features with
scale_pos_weight for class imbalance. Wrapped with CalibratedClassifierCV
(isotonic) for interpretable probabilities. Saved to models/lgbm_model.pkl.

**Feature Engineering (src/features.py)**
Stateless transforms applied identically at training and inference time.
TYPE_ENCODING is hardcoded to prevent inference-time LabelEncoder drift.
All 10 features are computed without lookahead.

**Redis Queue (docker-compose: redis service)**
Redis 7 Alpine. RQ workers subscribe to the "scoring" queue. Async jobs have
30s timeout and 300s result TTL. Decouples burst acceptance from processing.

**RQ Worker (src/queue_worker.py)**
Separate Docker container. Loads model once on startup (shared within the process).
Dequeues jobs, runs the full scoring pipeline, writes to PostgreSQL.

**PII Blinding (src/privacy.py)**
HMAC-SHA256 keyed on PII_SALT environment secret. Truncated to 16 hex chars.
One-way — audit log cannot be reversed without the secret. Compliant with
India's Digital Personal Data Protection Act 2023.

**PostgreSQL Audit Log (src/audit.py)**
SQLAlchemy ORM with QueuePool (pool_size=10, max_overflow=20, pool_pre_ping).
Append-only decisions table. Indexed on transaction_id. Every decision is
permanent — no update or delete operations exposed.

**EWMA Spike Monitor (src/monitor.py)**
Exponential weighted moving average of flagged-transaction rate over last 100
decisions. Alert threshold: 2x baseline fraud rate (2%). LLM-free.

**PSI Drift Monitor (src/drift.py)**
Population Stability Index between training feature distributions and recent
incoming transactions. PSI > 0.1 = warn. PSI > 0.2 = alert + retrain signal.
LLM-free. Reference stats saved at training time to models/reference_stats.json.

**Groq LLaMA Analyst Notes (src/analyst.py)**
llama-3.1-8b-instant via Groq API. Called only for REVIEW decisions, only after
the model has already decided. Never in the scoring path. Fails safely with
a fallback string if the API is unavailable.

**Monitoring Dashboard (src/dashboard.py + src/service.py)**
Single HTML page served at GET /dashboard. Fetches /dashboard/stats every 10s.
Shows: total decisions today, BLOCK/REVIEW/APPROVE distribution bar with percentages,
EWMA spike monitor with rate gauge, PSI drift monitor with per-feature bar chart,
PSI history timeline showing drift over time, last 20 decisions table.
In-memory PSI history (resets on restart — PostgreSQL persistence is future scope).

---

## Throughput Analysis

| Metric | Current | Target (production) |
|--------|---------|---------------------|
| Sync scoring latency (p50) | ~48ms (load test) / ~15ms (isolated) | <20ms |
| Sync scoring latency (p95) | ~540ms (100 users load) / ~35ms (isolated) | <50ms |
| Workers | 4 uvicorn processes | 8-16 behind nginx |
| Measured throughput (load test) | 219.89 TPS | 400-800 TPS |
| Load test config | 100 concurrent users, 60s, MacBook Air M1 | — |
| p50 latency (measured) | 48ms | <20ms |
| p95 latency (measured) | 540ms | <50ms |
| Async queue depth (burst) | Unbounded (Redis) | Unbounded (Redis/Kafka) |
| DB writes per decision | 1 (sync PostgreSQL) | 1 (async PostgreSQL) |
| DB pool size | 10 connections | 50 connections |
| Model size in memory | ~150MB per worker | Shared via Triton |

**Razorpay context:** Razorpay processes ~5-7M transactions per day (~58-80 TPS average,
400-800 TPS peak during festival sales). RazorSentry's sync path handles average load.
The async path handles peak bursts without blocking the payment gateway.

---

## Bottleneck Analysis

**Bottleneck 1: SHAP computation (~10-15ms per transaction)**
SHAP TreeExplainer runs on every scoring request. At 80 TPS this adds 800-1200ms
of cumulative SHAP work per second across all workers. Mitigation: cache SHAP
values for repeat transaction patterns, or move SHAP to an async annotation step
that runs after the decision is logged.

**Bottleneck 2: Model loaded 4 times (one per worker)**
Each of the 4 workers holds ~150MB of model in memory = ~600MB total just for
the model. Mitigation: Use Triton Inference Server or TorchServe to load the
model once and serve all workers via gRPC. This reduces memory and adds GPU
acceleration path.

**Bottleneck 3: Synchronous PostgreSQL writes**
Every /score call blocks until PostgreSQL acknowledges the write. At 80 TPS
this is 80 sequential write round-trips per second. Mitigation: buffer writes
in memory and flush in batches of 10-50, or write asynchronously after returning
the response (fire-and-forget with a retry queue for failures).

**Bottleneck 4: Single RQ worker**
One rq_worker container processes the async queue. If the queue depth grows
faster than one worker drains it, latency on async jobs grows. Mitigation:
scale rq_worker horizontally — each additional worker container independently
drains the same Redis queue with no coordination needed.

---

## Decision Policy

| Score | Decision | LLM Involved? | Next Step |
|-------|----------|---------------|-----------|
| ≥ 0.50 | BLOCK | No | Transaction rejected immediately |
| ≥ 0.35 (operating threshold) | REVIEW | Analyst note generated on demand | Human review within SLA |
| < 0.35 | APPROVE | No | Transaction proceeds |

Operating threshold (0.35) chosen by rupee net-savings maximisation on a
validation split. Never tuned on the test set.

---

## Privacy Design

| Data | Stored as | Reversible? | Where |
|------|-----------|-------------|-------|
| transaction_id | HMAC-SHA256[:16] | No (without PII_SALT) | PostgreSQL |
| nameOrig | HMAC-SHA256[:16] | No | PostgreSQL |
| nameDest | HMAC-SHA256[:16] | No | PostgreSQL |
| amount | Plaintext float | N/A | PostgreSQL |
| decision | Plaintext string | N/A | PostgreSQL |
| SHAP reasons | Plaintext JSON | N/A | PostgreSQL |

PII_SALT is an environment secret never committed to git.
Future scope: Hardware TEE (AWS Nitro Enclaves) for cryptographic attestation
of the entire scoring environment.

---

## Where We Chose NOT to Use an LLM

| Component | Why LLM was excluded |
|-----------|----------------------|
| Fraud scoring decision | Deterministic, auditable, sub-60ms required |
| SHAP reason codes | Mathematical — LLM interpretation would add hallucination risk |
| EWMA spike monitor | Statistical — speed and auditability matter more than language |
| PSI drift monitor | Statistical — PSI is a standard actuarial metric, needs no translation |
| Audit log writes | Correctness-critical — no place for probabilistic generation |
| Decision policy | Threshold is a number, not a prompt |

LLM is used in exactly one place: drafting a 2-line analyst note for REVIEW
transactions, after the decision is final and logged. This is a presentational
aid for human analysts, not a decision-making component.

---

## Future Scope

| Item | Description |
|------|-------------|
| Hardware TEE | AWS Nitro Enclaves for cryptographic attestation of scoring environment |
| Kafka | Replace Redis RQ for exactly-once delivery and message durability |
| Triton Inference Server | Load model once, serve all workers via gRPC, add GPU path |
| Kubernetes | Horizontal pod autoscaling on queue depth metric |
| WebSocket dashboard | Push-based updates instead of 10s polling |
| InfluxDB | Time-series storage for historical throughput and fraud rate trends |
| Model retraining pipeline | Triggered by PSI alert, CI/CD for model promotion |
| Per-merchant rate limiting | API keys instead of per-IP limits |
| Dashboard PSI history persistence | Store drift check results in PostgreSQL instead of in-memory — survives service restarts |
| Public deployment | Requires paid tier for PostgreSQL + Redis + app simultaneously — Railway ($5), Render ($7), or Fly.io ($5/month minimum for three services |
| One-command demo | `docker compose up --build -d` starts all four services locally — PostgreSQL, Redis, RazorSentry (4 workers), RQ worker |

---

## Model Card (Summary)

| Field | Value |
|-------|-------|
| Model | LightGBM + Isotonic calibration |
| Training data | PaySim — 500k legit + all fraud rows |
| Features | 10 (balance errors, drain flag, velocity, graph in-degree) |
| PR-AUC | 0.9999 (synthetic data — expected) |
| Precision @ threshold | 1.000 |
| Recall @ threshold | 0.9998 |
| Operating threshold | 0.35 (cost-optimised, calibrated) |
| Net savings on test set | ₹19.54 Cr across 101,643 transactions |
| False positive cost | ₹150/txn (review cost assumption) |
