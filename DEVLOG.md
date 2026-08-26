# RazorSentry — Dev Log

**Author:** Sayantan Mandal
**Started:** August 26, 2026

---

## Day 1 — Scaffold

- Repo initialized
- Folder structure created: `data/`, `models/`, `reports/`, `src/`, `tests/`, `scripts/`, `.github/workflows/`
- README, DEVLOG, ARCHITECTURE stubs written
- requirements.txt finalized with pinned versions
- Makefile, Dockerfile, docker-compose.yml configured
- CI workflow added for push/PR on main
- Source module stubs created: `features.py`, `train.py`, `eval.py`, `service.py`, `audit.py`
- Placeholder test added in `tests/test_service.py`

---

## Day 1 (continued) — Data Pipeline

- PaySim loaded from `data/PaySim.csv`
- Subsampled: all fraud rows + 500,000 random legit rows (random_state=42)
- Time-ordered split applied: first 80% → `data/train.parquet`, last 20% → `data/test.parquet`
- No shuffle at any stage — split boundary is strictly temporal on the `step` column
- Leakage note: all feature computation will be done strictly within split boundaries
- `src/data_loader.py` created with modular functions for load, subsample, sort, split, save, and summary
- `notebooks/eda.md` EDA template created
- Train fraud rate: 0.9762%
- Test fraud rate: 4.1754%

---

## Day 2 — Feature Engineering

- 10 features engineered
- Balance error features catch accounting inconsistencies
- Graph feature (`dest_in_degree_1h`) detects mule-account bursts without Neo4j (synthetic data has no graph ground truth)
- All features computed with no lookahead — leakage-safe

---

## Day 2-3 — Training & Cost Model

- Rules baseline: precision=0.1236, recall=0.4969
- LightGBM PR-AUC: 1.0000
- Operating threshold chosen by rupee net-savings, not F1
- Top 10 false positives saved honestly to `reports/top_fp_cases.csv`
- Leakage check: threshold tuned on validation split, not test set

---

## Day 4 — Audit Log

- SQLite append-only audit log via SQLAlchemy
- Every decision stored with UUID, score, decision, top-3 reasons, latency, model version
- No update or delete — immutable audit trail for regulators

---

## Day 4 — FastAPI Service

- POST /score with SHAP reason codes and audit logging
- POST /batch for batch replay demo
- GET /decisions/{id} for audit trail lookup
- GET /health for model version and threshold
- Decision boundary is 100% deterministic model — LLM never touches it

---

## Day 5 — LLM Analyst Notes + Spike Monitor

- LLM (Groq LLaMA — llama-3.1-8b-instant) drafts 2-line notes for REVIEW items only
- LLM never touches the decision — this is explicit in architecture
- Fraud spike monitor uses EWMA — deliberately LLM-free (fast, auditable)
- This is the "AI judgment" story: right tool in right place, and where we chose not to use one

---

## Day 5 — Eval Harness

- `make eval` reproduces every metric from scratch on held-out test set
- Sensitivity table shows assumptions are not cherry-picked
- Top-10 FP cases listed honestly — not hidden

---

## Day 6 — Batch Replay & Demo

- `batch_replay.py` streams 1000 test transactions live through the service
- `demo_curl.sh` has ready commands for video recording
- Demo shows one false positive landing in REVIEW (not BLOCK) — graceful failure handling

---

## Day 6 — Tests

- 6 pytest tests covering health, scoring, audit trail, batch, spike monitor
- Tests run in CI via GitHub Actions on every push
- No mocking of the model — tests use the actual loaded model
- Model-dependent tests degrade gracefully to 503 check when no artifact present in CI

---

## Day 7 — Final Polish

- README updated with metrics table and "What Broke" section
- Architecture doc finalized with Decision Policy table
- All files reviewed for clean code and one-line comments
- Submission ready

---

## Day 7 — Service Running + Tests
- FastAPI service started successfully on port 8000
- pytest results: 6 passed, 0 failed (100% pass)
- demo_curl.sh executed: high-risk CASH_OUT scored 0.8045 → REVIEW
- Groq analyst note working: yes

---

## Day 8 — Batch Replay + Final Submission Polish
- Batch replay: 1,000 transactions — 0 blocked, 5 review, 995 approved
- Avg latency: 3.2 ms
- BLOCK threshold corrected to 0.5 — three-tier policy now active across all decisions
- VIDEO_SCRIPT.md created
- All files pushed to GitHub
- Submission ready at https://github.com/Sayantan181222/RazorSentry


