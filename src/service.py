import os
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.analyst import generate_analyst_note
from src.audit import get_decision, get_recent_decisions, init_db, log_decision
from src.features import build_features, get_feature_columns
from src.monitor import check_fraud_spike

MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm_model.pkl")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "models/threshold.txt")
MODEL_VERSION = "lgbm_v1"
BLOCK_THRESHOLD = 0.5

REASON_MAP = {
    "balance_error_orig": "Accounting inconsistency in sender balance",
    "balance_error_dest": "Accounting inconsistency in receiver balance",
    "drain_flag": "Account being drained — amount near full balance",
    "zero_orig_after": "Sender account emptied to zero after transaction",
    "type_encoded": "Transaction type is high-risk",
    "amount_log": "Unusually large transaction amount",
    "orig_txn_count_1h": "Sender made multiple transactions in the last step",
    "orig_txn_sum_1h": "Sender moved large total volume in the last step",
    "dest_in_degree_1h": "Multiple senders hit this account in the last step",
    "high_amount_flag": "Transaction exceeds high-value amount threshold",
}

_model = None
_explainer = None
_operating_threshold: float = 0.5
_feature_columns: list[str] = []


# Loads the LightGBM model, SHAP explainer, threshold, and feature columns at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _explainer, _operating_threshold, _feature_columns
    init_db()
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        _explainer = shap.TreeExplainer(_model)
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r") as f:
            _operating_threshold = float(f.read().strip())
    _feature_columns = get_feature_columns()
    yield


app = FastAPI(title="RazorSentry", version="1.0.0", lifespan=lifespan)


# Pydantic schema for a single incoming transaction
class TransactionInput(BaseModel):
    transaction_id: str
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float


# Pydantic schema for a single scoring response
class ScoreResponse(BaseModel):
    decision_id: str
    transaction_id: str
    score: float
    decision: str
    reasons: list[str]
    latency_ms: float
    model_version: str


# Pydantic schema for the batch endpoint summary block
class BatchSummary(BaseModel):
    total: int
    blocked: int
    review: int
    approved: int


# Pydantic schema for the batch endpoint response
class BatchResponse(BaseModel):
    results: list[ScoreResponse]
    summary: BatchSummary


# Converts a TransactionInput Pydantic object to a single-row DataFrame
def _tx_to_df(tx: TransactionInput) -> pd.DataFrame:
    return pd.DataFrame([tx.model_dump()])


# Maps sorted SHAP magnitudes to the top N human-readable reason strings
def _top_reasons(shap_values: np.ndarray, feature_cols: list[str], top_n: int = 3) -> list[str]:
    pairs = sorted(
        zip(feature_cols, shap_values.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return [REASON_MAP.get(feat, feat) for feat, _ in pairs[:top_n]]


# Applies the three-tier decision policy based on score and thresholds
def _apply_policy(score: float) -> str:
    if score >= BLOCK_THRESHOLD:
        return "BLOCK"
    if score >= _operating_threshold:
        return "REVIEW"
    return "APPROVE"


# Scores a single transaction DataFrame row and returns all decision fields
def _score_transaction(tx: TransactionInput) -> ScoreResponse:
    t0 = time.perf_counter()

    df = _tx_to_df(tx)
    df_feat = build_features(df)
    X = df_feat[_feature_columns]

    prob = float(_model.predict_proba(X)[0, 1])
    decision = _apply_policy(prob)

    shap_vals = _explainer.shap_values(X)
    raw = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
    reasons = _top_reasons(raw, _feature_columns)

    latency_ms = (time.perf_counter() - t0) * 1000

    decision_id = log_decision(
        transaction_id=tx.transaction_id,
        score=prob,
        decision=decision,
        top_reasons=reasons,
        latency_ms=latency_ms,
        model_version=MODEL_VERSION,
        amount=tx.amount,
        transaction_type=tx.type,
    )

    return ScoreResponse(
        decision_id=decision_id,
        transaction_id=tx.transaction_id,
        score=round(prob, 6),
        decision=decision,
        reasons=reasons,
        latency_ms=round(latency_ms, 3),
        model_version=MODEL_VERSION,
    )


# Returns service liveness, loaded model version, operating threshold, and UTC timestamp
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "model_loaded": _model is not None,
        "operating_threshold": _operating_threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Scores a single transaction and writes the decision to the audit log
@app.post("/score", response_model=ScoreResponse)
def score(tx: TransactionInput) -> ScoreResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run src/train.py first")
    return _score_transaction(tx)


# Scores a batch of transactions sequentially and returns results with a summary
@app.post("/batch", response_model=BatchResponse)
def batch(transactions: list[TransactionInput]) -> BatchResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run src/train.py first")
    results = [_score_transaction(tx) for tx in transactions]
    summary = BatchSummary(
        total=len(results),
        blocked=sum(1 for r in results if r.decision == "BLOCK"),
        review=sum(1 for r in results if r.decision == "REVIEW"),
        approved=sum(1 for r in results if r.decision == "APPROVE"),
    )
    return BatchResponse(results=results, summary=summary)


# Retrieves a past decision from the audit log by its UUID
@app.get("/decisions/{decision_id}")
def fetch_decision(decision_id: str) -> dict:
    record = get_decision(decision_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")
    return record


# Returns the most recent N decisions from the audit log
@app.get("/decisions")
def recent_decisions(limit: int = 50) -> list[dict]:
    return get_recent_decisions(limit=limit)


# Fetches the last 100 decisions, extracts scores, and returns an EWMA spike alert
@app.get("/monitor/spike")
def monitor_spike() -> dict:
    decisions = get_recent_decisions(limit=100)
    scores = [d["score"] for d in decisions if "score" in d]
    return check_fraud_spike(scores, window=100, threshold=_operating_threshold)


# Generates a 2-line LLM analyst note for a REVIEW-queue transaction by decision ID
@app.get("/analyst/note/{decision_id}")
def analyst_note(decision_id: str) -> dict:
    record = get_decision(decision_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")
    if record["decision"] != "REVIEW":
        raise HTTPException(
            status_code=400,
            detail="Analyst notes are only generated for REVIEW decisions",
        )
    note = generate_analyst_note(
        transaction=record,
        score=record["score"],
        reasons=record["top_reasons"],
    )
    return {"decision_id": decision_id, "analyst_note": note}
