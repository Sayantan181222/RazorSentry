import os
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import shap
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator
from redis import Redis
from rq import Queue
from rq.job import Job, NoSuchJobError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.analyst import generate_analyst_note
from src.audit import check_db_health, get_decision, get_recent_decisions, init_db, log_decision
from src.drift import check_drift
from src.features import build_features, get_feature_columns
from src.monitor import check_fraud_spike
from src.privacy import blind_identifier, blind_transaction

MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm_model.pkl")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "models/threshold.txt")
MODEL_VERSION = "lgbm_v1"
BLOCK_THRESHOLD = 0.5
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_redis_conn: Redis | None = None
_score_queue: Queue | None = None

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


# Initialises Redis connection and RQ queue, returns False if Redis is unavailable
def _init_redis() -> bool:
    global _redis_conn, _score_queue
    try:
        _redis_conn = Redis.from_url(REDIS_URL, socket_connect_timeout=2)
        _redis_conn.ping()
        _score_queue = Queue("scoring", connection=_redis_conn)
        return True
    except Exception:
        return False


# Loads the LightGBM model, SHAP explainer, threshold, and feature columns at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _explainer, _operating_threshold, _feature_columns
    _init_redis()
    import time as _time
    for attempt in range(10):
        try:
            init_db()
            break
        except Exception as e:
            if attempt == 9:
                raise RuntimeError(f"Database unavailable after 10 attempts: {e}")
            _time.sleep(3)
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        tree_model = _model.estimator if hasattr(_model, "estimator") else (_model.calibrated_classifiers_[0].estimator if hasattr(_model, "calibrated_classifiers_") else _model)
        _explainer = shap.TreeExplainer(tree_model)
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r") as f:
            _operating_threshold = float(f.read().strip())
    _feature_columns = get_feature_columns()
    yield


# Rate limiter — 60 score requests per minute per IP
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app = FastAPI(title="RazorSentry", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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

    model_config = {"str_strip_whitespace": True}

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("amount must be non-negative")
        return v

    @field_validator("type")
    @classmethod
    def type_must_be_valid(cls, v: str) -> str:
        valid = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
        if v.upper() not in valid:
            raise ValueError(f"type must be one of {valid}")
        return v.upper()

    @field_validator("step")
    @classmethod
    def step_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("step must be >= 1")
        return v


class RazorpayWebhookPayload(BaseModel):
    entity: str = "event"
    account_id: str
    event: str
    contains: list
    payload: dict


class ScoreResponse(BaseModel):
    decision_id: str
    transaction_id: str
    score: float
    decision: str
    reasons: list[str]
    latency_ms: float
    model_version: str


class BatchSummary(BaseModel):
    total: int
    blocked: int
    review: int
    approved: int


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

    blinded_txn_id = blind_identifier(tx.transaction_id)
    decision_id = log_decision(
        transaction_id=blinded_txn_id,
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


# Returns 200 only when model is loaded and database is reachable — used by Docker healthcheck
@app.get("/ready")
def ready() -> dict:
    from src.audit import check_db_health
    db_ok = check_db_health()
    model_ok = _model is not None
    if not db_ok or not model_ok:
        raise HTTPException(
            status_code=503,
            detail={
                "ready": False,
                "model_loaded": model_ok,
                "db_reachable": db_ok,
            }
        )
    return {
        "ready": True,
        "model_loaded": model_ok,
        "db_reachable": db_ok,
    }


# Scores a single transaction and writes the decision to the audit log
@app.post("/score", response_model=ScoreResponse)
@limiter.limit("60/minute")
def score(request: Request, tx: TransactionInput) -> ScoreResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run src/train.py first")
    return _score_transaction(tx)


# Scores a batch of transactions sequentially and returns results with a summary
@app.post("/batch", response_model=BatchResponse)
@limiter.limit("10/minute")
def batch(request: Request, transactions: list[TransactionInput]) -> BatchResponse:
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


# Maps Razorpay payment method strings to PaySim transaction type strings
def _map_razorpay_method(method: str) -> str:
    mapping = {
        "card": "PAYMENT",
        "netbanking": "TRANSFER",
        "wallet": "CASH_OUT",
        "upi": "PAYMENT",
        "emi": "PAYMENT",
    }
    return mapping.get(method.lower(), "PAYMENT")


# Accepts a Razorpay-shaped webhook and routes it through the fraud scoring pipeline
@app.post("/webhook/razorpay")
async def razorpay_webhook(webhook: RazorpayWebhookPayload):
    try:
        payment = webhook.payload.get("payment", {}).get("entity", {})
        txn = TransactionInput(
            transaction_id=payment.get("id", "rzp_unknown"),
            step=1,
            type=_map_razorpay_method(payment.get("method", "PAYMENT")),
            amount=float(payment.get("amount", 0)) / 100,
            nameOrig=payment.get("contact", "C_UNKNOWN"),
            oldbalanceOrg=0.0,
            newbalanceOrig=0.0,
            nameDest=payment.get("email", "M_UNKNOWN"),
            oldbalanceDest=0.0,
            newbalanceDest=0.0,
        )
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded — run src/train.py first")
        result = _score_transaction(txn)
        return {
            "webhook_event": webhook.event,
            "razorpay_payment_id": payment.get("id"),
            "razorsentry_decision": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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


# Checks PSI drift between recent scored transactions and training distribution
@app.get("/monitor/drift")
def monitor_drift() -> dict:
    decisions = get_recent_decisions(limit=200)
    if len(decisions) < 10:
        return {"drift_checked": False, "reason": "Not enough recent decisions for drift check"}
    rows = []
    for d in decisions:
        rows.append({
            "amount": d.get("amount", 0),
            "transaction_type": d.get("transaction_type", "PAYMENT"),
        })
    df = pd.DataFrame(rows)
    df["amount_log"] = np.log1p(df["amount"])
    df["high_amount_flag"] = (df["amount"] > 200000).astype(int)
    from src.features import get_feature_columns
    available_cols = [c for c in ["amount_log", "high_amount_flag"] if c in df.columns]
    return check_drift(df, available_cols)


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


# Enqueues a transaction for async scoring and immediately returns a job_id
@app.post("/score/async")
@limiter.limit("200/minute")
def score_async(request: Request, tx: TransactionInput) -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if _score_queue is None:
        raise HTTPException(
            status_code=503,
            detail="Queue unavailable — falling back to /score for sync scoring"
        )
    from src.queue_worker import score_transaction_job
    job = _score_queue.enqueue(
        score_transaction_job,
        tx.model_dump(),
        job_timeout=30,
        result_ttl=300,
    )
    return {
        "job_id": job.id,
        "status": "queued",
        "poll_url": f"/score/result/{job.id}",
        "message": "Transaction queued for scoring. Poll poll_url for result.",
    }


# Polls Redis for the result of an async scoring job by job_id
@app.get("/score/result/{job_id}")
def score_result(job_id: str) -> dict:
    if _redis_conn is None:
        raise HTTPException(status_code=503, detail="Queue unavailable")
    try:
        job = Job.fetch(job_id, connection=_redis_conn)
    except NoSuchJobError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.is_finished:
        return {"job_id": job_id, "status": "complete", "result": job.result}
    if job.is_failed:
        return {"job_id": job_id, "status": "failed", "error": str(job.exc_info)}
    if job.is_started:
        return {"job_id": job_id, "status": "processing"}
    return {"job_id": job_id, "status": "queued"}
