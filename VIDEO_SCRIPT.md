# RazorSentry — 5 Min Pitch Cue Card
Author: Sayantan Mandal, Gati Shakti Vishwavidyalaya

## 0:00–0:30 | The Problem
- Fraud detection is not a Kaggle contest — blocking a legit customer costs money too
- Two costs exist: the fraud loss AND the false positive review cost at ₹150/txn
- Most systems optimise F1 — RazorSentry optimises rupee net savings

## 0:30–2:15 | Live Demo
- Run: bash scripts/demo_curl.sh
- Show the BLOCK decision with reason codes on the CASH_OUT transaction
- Show the APPROVE decision — good customer, no friction added
- Run batch_replay.py — show decisions streaming live in terminal
- Point out: one false positive lands in REVIEW queue, not a hard BLOCK — graceful handling

## 2:15–3:15 | Metrics
- Open reports/metrics.json — show PR-AUC, precision, recall at operating threshold
- Show reports/cost_curve.png — "this is where we picked our threshold, not at F1"
- Show reports/top_fp_cases.csv — "these are the 10 good customers we got wrong — not hidden"
- Show sensitivity table — "savings hold even if FP cost assumptions change"

## 3:15–4:15 | Architecture and AI Judgment
- LightGBM decides in under 20ms — auditable, bounded, explainable
- SHAP gives reason codes in plain English: "Multiple senders hit this account in the last hour"
- Groq LLaMA drafts a 2-line analyst note for REVIEW items only — never touches the decision
- Spike monitor uses EWMA — deliberately LLM-free: fast, auditable, no hallucination risk
- Show GET /decisions/{id} — full immutable audit trail per transaction

## 4:15–5:00 | What Broke and What is Next
- Velocity features leaked future data — caught it, fixed by enforcing split boundaries
- LabelEncoder was re-fit at inference — wrong encoding every time — fixed with hardcoded TYPE_ENCODING
- drain_flag fired on zero-balance accounts — fixed with non-zero guard
- Threshold-on-test would be cheating — tuned on validation split only
- What is next: Razorpay webhook integration, real-time streaming features
