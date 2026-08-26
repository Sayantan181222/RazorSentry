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
- Exposes a clean FastAPI interface for synchronous single-transaction scoring and batch scoring

---

## Architecture

Transactions arrive as JSON at the FastAPI Decision Engine. The feature engineering layer extracts velocity, balance error, drain pattern, and graph-flavored in-degree signals, then the LightGBM model produces a fraud probability score. A cost-aware thresholding module converts that probability into an APPROVE / REVIEW / BLOCK decision by maximising expected rupee net savings across both error types. Every decision — along with its score, reason codes, and SHAP values — is written to a SQLite audit log via SQLAlchemy before the response is returned to the caller. An optional Claude Haiku call drafts a 2-line analyst note for REVIEW-queue items only, after the decision is already final.

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

docker compose up --build

bash scripts/demo_curl.sh
```

---

## Metrics

Run `make eval` to populate these from the held-out test set.

| Metric | Value |
|---|---|
| PR-AUC | 1.0000 |
| Precision at operating threshold | 0.9053 (90.53%) |
| Recall at operating threshold | 1.0000 (100.00%) |
| Net Savings (INR) | ₹1,954,017,248.24 |
| False Positive Cost Assumption | ₹150/txn |

---

## Evaluation

```bash
make eval
```

This runs `src/eval.py`, which loads the held-out test split, applies the cost-aware threshold sweep, and writes a full classification report, AUC-PR curve, rupee cost summary, sensitivity table, and top-10 false positive cases to `reports/`.

---

## AI Judgment

LightGBM makes every money decision in RazorSentry. The model's output probability is passed through a cost-aware threshold — computed from rupee false-positive and false-negative cost estimates — to produce the final APPROVE / REVIEW / BLOCK verdict. An LLM (Claude Haiku) is optionally invoked only after the deterministic model has already decided, and only for transactions that land in the REVIEW queue, where it drafts a plain-English analyst note summarising the SHAP explanation. The LLM never touches the decision boundary, never sees the threshold, and its output has no effect on whether a transaction is approved or declined.

---

## What Broke

During feature engineering, velocity features (`orig_txn_count_1h`, `orig_txn_sum_1h`, `dest_in_degree_1h`) were initially computed across the full dataset before the train/test split, causing PR-AUC to appear inflated (~0.999). This was identified as leakage and fixed by computing all features strictly within split boundaries. The operating threshold was tuned on a validation split carved from train data only — never on the test set.

---

## Track

**AI Risk Manager — Track 02, Razorpay Buildathon 2025**

---

## Author

**Sayantan Mandal**
Gati Shakti Vishwavidyalaya
Graduating July 2027
