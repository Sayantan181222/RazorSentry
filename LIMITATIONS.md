# RazorSentry — Known Limitations & Future Scope

## Current Limitations

### 1. Synthetic Data Gap
RazorSentry is trained on [PaySim](https://www.kaggle.com/datasets/mtalaltariq/paysim-data),
a synthetic mobile-money simulator. PaySim encodes fraud via deterministic
accounting anomalies — real-world fraud is noisier and more adversarial.
Near-perfect metrics (PR-AUC 0.9999) are a property of the synthetic dataset,
not a claim about real-world performance. **Retraining on live transaction data
is required before production use.**

### 2. In-Memory PSI History
The PSI drift history shown in the dashboard is stored in-memory inside the
FastAPI process. It resets every time the service restarts. In production,
drift check results should be written to PostgreSQL with timestamps so long-term
model health trends are queryable.

### 3. SHAP CPU Latency Under Load
SHAP TreeExplainer runs on every /score request to generate reason codes.
Under sustained load (100+ concurrent users), SHAP adds 10-15ms per request
and becomes the primary bottleneck. p95 latency rises to 540ms at 100 concurrent
users. The production mitigation is async SHAP annotation — log the decision
first, compute SHAP after, and push reason codes to the audit record
asynchronously.

### 4. Single RQ Worker
The async scoring path uses one RQ worker container. If queue depth grows
faster than one worker drains it, async job latency grows unboundedly.
Mitigation: scale rq_worker horizontally — each additional worker container
drains the same Redis queue independently.

### 5. Velocity Features Use Step=1 Window
Velocity features (orig_txn_count_1h, orig_txn_sum_1h, dest_in_degree_1h)
use a window of one PaySim step. This may miss slow-burn fraud patterns
that develop over many hours. A sliding window over real timestamps would
be more robust on live data.

---

## Future Scope

| Item | Description | Priority |
|------|-------------|----------|
| Retrain on live data | Calibrate model on real Razorpay transaction data | High |
| Triton Inference Server | Load model once, serve all workers via gRPC | High |
| Kafka | Replace Redis RQ for exactly-once delivery and durability | High |
| PostgreSQL PSI persistence | Store drift check history for long-term model health | Medium |
| Async SHAP annotation | Decouple SHAP from the scoring hot path | Medium |
| Hardware TEE | AWS Nitro Enclaves for cryptographic attestation of scoring | Medium |
| Kubernetes HPA | Autoscale on queue depth metric | Medium |
| Per-merchant rate limiting | API keys instead of per-IP rate limits | Low |
| WebSocket dashboard | Push-based updates instead of 10s polling | Low |

---

## Honest Metrics Note

All numbers in this repo (PR-AUC, precision, recall, net savings) are
computed on the PaySim held-out test set using a time-ordered split.
They are honest within the synthetic data context. They are not
cherry-picked — the top 10 false positive cases are committed to
`reports/top_fp_cases.csv` and the sensitivity table shows results
hold across different false-positive cost assumptions.
