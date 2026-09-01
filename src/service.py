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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator
from redis import Redis
from rq import Queue
from rq.job import Job, NoSuchJobError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.analyst import generate_analyst_note
from src.audit import check_db_health, get_decision, get_pool_stats, get_recent_decisions, init_db, log_decision
from src.dashboard import get_dashboard_stats
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


# Rate limiter — 60 score requests per minute per IP (disabled in LOAD_TEST_MODE)
LOAD_TEST_MODE = os.getenv("LOAD_TEST_MODE", "false").lower() == "true"
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[] if LOAD_TEST_MODE else ["60/minute"],
    enabled=not LOAD_TEST_MODE,
)

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


# Returns 200 only when model is loaded and database is reachable — used by Docker and Kubernetes healthchecks
@app.get("/ready")
def ready() -> dict:
    from src.audit import check_db_health
    import time
    t0 = time.perf_counter()
    db_ok = check_db_health()
    db_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    model_ok = _model is not None
    threshold_ok = _operating_threshold is not None
    redis_ok = _redis_conn is not None
    try:
        if redis_ok:
            _redis_conn.ping()
    except Exception:
        redis_ok = False
    all_ok = model_ok and db_ok
    if not all_ok:
        raise HTTPException(
            status_code=503,
            detail={
                "ready": False,
                "model_loaded": model_ok,
                "db_reachable": db_ok,
                "db_latency_ms": db_latency_ms,
                "threshold_loaded": threshold_ok,
                "redis_reachable": redis_ok,
            }
        )
    return {
        "ready": True,
        "model_loaded": model_ok,
        "db_reachable": db_ok,
        "db_latency_ms": db_latency_ms,
        "threshold_loaded": threshold_ok,
        "redis_reachable": redis_ok,
        "operating_threshold": _operating_threshold,
        "model_version": MODEL_VERSION,
    }


# Returns PostgreSQL connection pool statistics for infrastructure monitoring
@app.get("/health/pool")
def pool_health() -> dict:
    return {
        "pool_stats": get_pool_stats(),
        "model_version": MODEL_VERSION,
        "workers": 4,
        "note": "pool stats are per-worker — multiply checked_out by workers for total connections"
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


# Computes PSI drift between recent scored transactions and training distribution
@app.get("/monitor/drift")
def monitor_drift() -> dict:
    decisions = get_recent_decisions(limit=200)
    if len(decisions) < 10:
        return {"drift_checked": False, "reason": "Not enough recent decisions for drift check"}
    import numpy as np
    rows = []
    for d in decisions:
        amount = d.get("amount", 0)
        rows.append({
            "amount": amount,
            "amount_log": float(np.log1p(amount)),
            "high_amount_flag": int(amount > 200000),
            "large_amount_flag": int(amount > 500000),
            "score": d.get("score", 0),
        })
    df = pd.DataFrame(rows)
    available_cols = [c for c in df.columns if c != "amount"]
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


# Returns a JSON stats payload for the dashboard frontend to consume
@app.get("/dashboard/stats")
def dashboard_stats() -> dict:
    from src.audit import check_db_health
    recent = get_recent_decisions(limit=200)
    stats = get_dashboard_stats(recent)
    last_10 = recent[:20]
    scores = [d["score"] for d in recent if "score" in d]
    spike = check_fraud_spike(scores, window=100, threshold=_operating_threshold)
    if len(recent) < 10:
        drift = {"drift_checked": False, "reason": "Insufficient data"}
    else:
        import numpy as np
        rows = []
        for d in recent:
            amount = d.get("amount", 0)
            rows.append({
                "amount": amount,
                "amount_log": float(np.log1p(amount)),
                "high_amount_flag": int(amount > 200000),
                "large_amount_flag": int(amount > 500000),
                "score": d.get("score", 0),
            })
        df = pd.DataFrame(rows)
        available_cols = [c for c in df.columns if c != "amount"]
        drift = check_drift(df, available_cols)
        from src.dashboard import record_drift_history
        record_drift_history(drift)
    from src.dashboard import get_drift_history
    return {
        "stats": stats,
        "last_10_decisions": last_10,
        "spike_alert": spike,
        "drift_alert": drift,
        "drift_history": get_drift_history(),
        "db_healthy": check_db_health(),
        "model_version": MODEL_VERSION,
        "operating_threshold": _operating_threshold,
    }


# Serves the live monitoring dashboard HTML page
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RazorSentry — Fraud Operations Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #F3F5F9;
  --card: #FFFFFF;
  --line: #E8ECF2;
  --line2: #F0F3F8;
  --track: #EDEFF4;
  --text: #101828;
  --text2: #5B6B85;
  --text3: #98A2B3;
  --green: #12B76A;   --greenD: #079451;  --greenT: #E9F9F0;
  --red: #F04438;     --redD: #B42318;    --redT: #FEF3F2;
  --amber: #F79009;   --amberD: #B54708;  --amberT: #FEF0E6;
  --indigo: #5A5AE6;  --indigoD: #444CE7; --indigoT: #EEF0FF;
  --violet: #8B5CF6;  --violetD: #6941C6; --violetT: #F4F0FF;
  --sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --mono: 'JetBrains Mono', 'SF Mono', 'Cascadia Code', Consolas, monospace;
  --brand: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  font-family: var(--sans); background: var(--bg); color: var(--text);
  min-height: 100vh; padding: 20px clamp(12px, 2.5vw, 32px);
  -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}
::selection { background: rgba(90,90,230,.18); }
.mono { font-family: var(--mono); font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
.ic { width: 14px; height: 14px; stroke: currentColor; fill: none; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; flex: none; }

/* ---------- Entrance animations ---------- */
@keyframes fadeUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
.pagehead  { animation: fadeUp .5s cubic-bezier(.22,1,.36,1) backwards; }
.sidebar   { animation: fadeUp .6s cubic-bezier(.22,1,.36,1) backwards; }
.card      { animation: fadeUp .6s cubic-bezier(.22,1,.36,1) backwards; }
.kpis .card:nth-child(1) { animation-delay: .06s; }
.kpis .card:nth-child(2) { animation-delay: .12s; }
.kpis .card:nth-child(3) { animation-delay: .18s; }
.kpis .card:nth-child(4) { animation-delay: .24s; }
.kpis .card:nth-child(5) { animation-delay: .30s; }
.grid-2  .card:nth-child(1) { animation-delay: .36s; }
.grid-2  .card:nth-child(2) { animation-delay: .42s; }
.grid-2b .card:nth-child(1) { animation-delay: .48s; }
.grid-2b .card:nth-child(2) { animation-delay: .54s; }
#panel-decisions { animation-delay: .62s; }
.foot { animation: fadeUp .5s .7s cubic-bezier(.22,1,.36,1) backwards; }

.app { max-width: 1320px; margin: 0 auto; display: flex; gap: 20px; align-items: flex-start; }

/* ---------- Sidebar ---------- */
.sidebar { width: 232px; flex: none; background: var(--card); border: 1px solid var(--line);
  border-radius: 16px; padding: 18px 14px; position: sticky; top: 20px;
  box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 10px 28px -16px rgba(16,24,40,.1);
  display: flex; flex-direction: column; gap: 20px; }
.sb-logo { display: flex; align-items: center; gap: 11px; padding: 2px 6px 14px; border-bottom: 1px solid var(--line2); }
.sb-name { font-family: var(--brand); font-size: 1.05rem; font-weight: 700; letter-spacing: -.01em; }
.sb-tag { font-size: .6rem; color: var(--text3); margin-top: 1px; font-weight: 500; }
.sb-nav { display: flex; flex-direction: column; gap: 3px; }
.sb-link { display: flex; align-items: center; gap: 11px; padding: 9px 10px; border-radius: 10px;
  font-size: .74rem; font-weight: 600; color: var(--text2); text-decoration: none;
  transition: background .16s, color .16s, transform .16s; }
.sb-link:hover { background: #F2F5FB; color: var(--text); transform: translateX(3px); }
.sb-link.active { background: var(--indigoT); color: var(--indigoD); }
.sb-link .ic { color: var(--text3); }
.sb-link.active .ic, .sb-link:hover .ic { color: var(--indigoD); }
.sb-sys { margin-top: auto; background: #F8FAFC; border: 1px solid var(--line2); border-radius: 12px; padding: 12px; }
.sb-sys-title { font-size: .56rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; color: var(--text3); margin-bottom: 9px; }
.sys-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; font-size: .67rem; padding: 4px 0; }
.sys-k { color: var(--text2); }
.sys-v { color: var(--text); font-weight: 600; }
.sys-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); display: inline-block; }
.sys-dot.bad { background: var(--red); }

/* ---------- Main ---------- */
.main { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 18px; }
.mobile-brand { display: none; }
.ph-title { font-family: var(--brand); font-size: 1.35rem; font-weight: 700; letter-spacing: -.01em; }
.ph-sub { display: flex; align-items: center; gap: 8px; font-size: .72rem; color: var(--text2); margin-top: 5px; flex-wrap: wrap; font-weight: 500; }
.livedot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); animation: liveRing 2.2s ease-out infinite; }
.livedot.bad { background: var(--red); animation: liveRingBad 2.2s ease-out infinite; }
@keyframes liveRing { 0% { box-shadow: 0 0 0 0 rgba(18,183,106,.4); opacity: 1; } 70% { box-shadow: 0 0 0 9px rgba(18,183,106,0); opacity: .55; } 100% { box-shadow: 0 0 0 0 rgba(18,183,106,0); opacity: 1; } }
@keyframes liveRingBad { 0% { box-shadow: 0 0 0 0 rgba(240,68,56,.4); opacity: 1; } 70% { box-shadow: 0 0 0 9px rgba(240,68,56,0); opacity: .55; } 100% { box-shadow: 0 0 0 0 rgba(240,68,56,0); opacity: 1; } }
.livetxt { color: var(--greenD); font-weight: 700; letter-spacing: .13em; font-size: .58rem; }
.ph-right { display: flex; align-items: center; gap: 10px; }
.pill { display: inline-flex; align-items: center; gap: 6px; font-size: .66rem; font-weight: 600;
  padding: 6px 12px; border-radius: 999px; background: var(--card); border: 1px solid var(--line); color: var(--text2);
  transition: border-color .2s, box-shadow .2s; }
.pill:hover { border-color: rgba(90,90,230,.35); box-shadow: 0 4px 14px -6px rgba(67,76,230,.35); }
.pill .ic { width: 12px; height: 12px; color: var(--indigo); }

/* ---------- Cards + LIGHT HOVER ---------- */
.card { background: var(--card); border: 1px solid var(--line); border-radius: 16px;
  box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 10px 28px -18px rgba(16,24,40,.1);
  transition: background .25s ease, border-color .25s ease, transform .25s cubic-bezier(.34,1.4,.64,1), box-shadow .25s ease; }
.main .card:hover {
  background: linear-gradient(180deg, #F5F7FF 0%, #FFFFFF 65%);   /* light indigo tint */
  border-color: rgba(90,90,230,.30);
  transform: translateY(-3px);
  box-shadow: 0 2px 4px rgba(16,24,40,.04), 0 22px 44px -16px rgba(67,76,230,.25);
}
#panel-decisions:hover { transform: none; }

/* ---------- KPI ---------- */
.kpis { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; }
.kpi { padding: 16px; display: flex; flex-direction: column; gap: 10px; }
.kpi-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; }
.kpi-ic { width: 34px; height: 34px; border-radius: 10px; display: inline-flex; align-items: center; justify-content: center; flex: none;
  transition: transform .25s cubic-bezier(.34,1.56,.64,1); }
.kpi:hover .kpi-ic { transform: scale(1.12) rotate(-4deg); }
.kpi-ic .ic { width: 16px; height: 16px; }
.ti-indigo { background: var(--indigoT); color: var(--indigo); }
.ti-red    { background: var(--redT);    color: var(--red); }
.ti-amber  { background: var(--amberT);  color: var(--amber); }
.ti-green  { background: var(--greenT);  color: var(--green); }
.ti-violet { background: var(--violetT); color: var(--violet); }
.kpi-label { font-size: .62rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--text2); padding-top: 8px; }
.kpi-value { font-family: var(--brand); font-size: 1.85rem; font-weight: 700; letter-spacing: -.02em; line-height: 1;
  font-variant-numeric: tabular-nums; transform-origin: left center; }
.kpi-value.flash { animation: vflash .6s cubic-bezier(.34,1.56,.64,1); }
@keyframes vflash { 30% { transform: scale(1.07); } }
.v-navy { color: var(--text); } .v-red { color: var(--redD); } .v-amber { color: var(--amberD); }
.v-green { color: var(--greenD); } .v-violet { color: var(--violetD); }
.kpi-foot { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-top: auto; }
.spark { width: 100%; height: 34px; display: block; }

/* delta pills — pop in on update */
.dp { display: inline-flex; align-items: center; gap: 4px; font-size: .64rem; font-weight: 700;
  padding: 3px 9px; border-radius: 999px; font-variant-numeric: tabular-nums; white-space: nowrap;
  animation: popIn .35s cubic-bezier(.34,1.56,.64,1); }
@keyframes popIn { from { transform: scale(.6); opacity: 0; } }
.dp.up   { color: var(--greenD); background: var(--greenT); }
.dp.down { color: var(--redD);   background: var(--redT); }
.dp.flat { color: var(--text2);  background: #F2F4F7; }

/* ---------- Panels ---------- */
.grid-2 { display: grid; grid-template-columns: 1.12fr 1fr; gap: 16px; }
.grid-2b { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.panel { padding: 18px 20px; scroll-margin-top: 24px; }
.panel-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 14px; }
.panel-title { display: inline-flex; align-items: center; gap: 8px; font-size: .66rem; font-weight: 700;
  letter-spacing: .11em; text-transform: uppercase; color: var(--text); }
.panel-title .ic { color: var(--indigo); }
.panel-aux { font-size: .63rem; color: var(--text3); }
.panel-sub { font-size: .7rem; color: var(--text2); line-height: 1.55; font-weight: 500; }
.empty { padding: 18px 0; text-align: center; color: var(--text3); font-size: .72rem; }

.chip { display: inline-flex; align-items: center; gap: 6px; font-size: .58rem; font-weight: 700;
  letter-spacing: .09em; padding: 4px 11px; border-radius: 999px; text-transform: uppercase; }
.chip .cdot { width: 6px; height: 6px; border-radius: 50%; }
.chip.ok { color: var(--greenD); background: var(--greenT); border: 1px solid rgba(18,183,106,.3); }
.chip.ok .cdot { background: var(--green); }
.chip.warn { color: var(--amberD); background: var(--amberT); border: 1px solid rgba(247,144,9,.3); }
.chip.warn .cdot { background: var(--amber); }
.chip.danger { color: var(--redD); background: var(--redT); border: 1px solid rgba(240,68,56,.32); }
.chip.danger .cdot { background: var(--red); animation: cdotBlink 1.4s infinite; }
@keyframes cdotBlink { 0%,100% { opacity: 1; } 50% { opacity: .3; } }

/* ---------- Distribution ---------- */
.dist-layout { display: flex; align-items: center; gap: 26px; }
.donut-wrap { position: relative; width: 150px; height: 150px; flex: none; }
.donut { width: 150px; height: 150px; display: block; }
.dn-track { fill: none; stroke: var(--track); stroke-width: 14; }
.dn-seg { fill: none; stroke-width: 14; transition: stroke-width .22s ease; animation: segIn .5s ease backwards; }
.dn-seg:hover { stroke-width: 18; }
@keyframes segIn { from { opacity: 0; } }
.donut-center { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 2px; }
.dc-val { font-family: var(--brand); font-size: 1.5rem; font-weight: 700; font-variant-numeric: tabular-nums; color: var(--text); }
.dc-lab { font-size: .55rem; letter-spacing: .16em; color: var(--text3); text-transform: uppercase; font-weight: 600; }
.dist-side { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 12px; }
.dist-bar { display: flex; height: 12px; border-radius: 7px; overflow: hidden; background: var(--track); }
.dist-block { background: linear-gradient(180deg,#F97066,#F04438); }
.dist-review { background: linear-gradient(180deg,#FDB022,#F79009); }
.dist-approve { background: linear-gradient(180deg,#3CC97C,#12B76A); }
.dist-legend { display: flex; gap: 18px; flex-wrap: wrap; }
.lg { display: inline-flex; align-items: center; gap: 7px; font-size: .72rem; color: var(--text2); font-weight: 500; }
.lgd { width: 9px; height: 9px; border-radius: 3px; }
.lg b { font-weight: 700; color: var(--text); }
.dist-note { font-size: .68rem; color: var(--text3); font-weight: 500; }

/* ---------- Spike monitor ---------- */
.spike-hero { display: flex; align-items: baseline; gap: 10px; margin-bottom: 12px; }
.ewma-val { font-family: var(--brand); font-size: 1.9rem; font-weight: 700; letter-spacing: -.02em; font-variant-numeric: tabular-nums; color: var(--text); }
.ewma-lab { font-size: .63rem; color: var(--text2); font-weight: 500; }
.meter { margin: 2px 0 12px; }
.meter-track { position: relative; height: 12px; border-radius: 7px; background: var(--track); overflow: hidden; }
.meter-fill { height: 100%; border-radius: 7px; background: var(--green); transition: width .6s cubic-bezier(.22,1,.36,1), background .3s; }
.mtick { position: absolute; top: 0; bottom: 0; width: 1px; background: rgba(16,24,40,.14); }
.meter-scale { display: flex; justify-content: space-between; font-size: .57rem; color: var(--text3); margin-top: 6px; }

/* ---------- Drift bars ---------- */
.drift-meta { font-size: .68rem; color: var(--text2); margin-bottom: 12px; }
.pb-row { display: flex; align-items: center; gap: 10px; padding: 5px 0; }
.pb-name { width: 128px; font-size: .66rem; color: var(--text2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: none; }
.pb-track { display: block; position: relative; flex: 1; height: 8px; border-radius: 5px; background: var(--track); }
.pb-fill { display: block; height: 100%; border-radius: 5px; }
.pb-tick { position: absolute; top: -3px; bottom: -3px; width: 1px; background: rgba(16,24,40,.18); }
.pb-val { width: 72px; text-align: right; font-size: .66rem; flex: none; font-weight: 700; }
.psi-note { margin-top: 10px; font-size: .6rem; color: var(--text3); }

/* ---------- PSI history — SCROLLABLE ---------- */
.psi-scroll { max-height: 340px; overflow-y: auto; overflow-x: hidden; padding-right: 8px; }
.tl-row { display: flex; align-items: center; gap: 9px; padding: 7px 4px; border-radius: 8px;
  border-bottom: 1px solid var(--line2); transition: background .15s, transform .15s; }
.tl-row:hover { background: #F7F9FC; transform: translateX(2px); }
.tl-row:last-child { border-bottom: none; }
.tl-time { width: 58px; font-size: .64rem; color: var(--text2); flex: none; }
.tl-dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.tl-dot.ok { background: var(--green); }
.tl-dot.warn { background: var(--amber); }
.tl-dot.alert { background: var(--red); }
.tl-track { display: block; flex: 1; height: 6px; border-radius: 4px; background: var(--track); overflow: hidden; }
.tl-fill { display: block; height: 100%; border-radius: 4px; }
.tl-psi { width: 52px; text-align: right; font-size: .66rem; font-weight: 700; flex: none; }
.tl-n { width: 50px; text-align: right; font-size: .6rem; color: var(--text3); flex: none; }

/* light scrollbars */
.slim::-webkit-scrollbar { width: 8px; height: 8px; }
.slim::-webkit-scrollbar-track { background: transparent; }
.slim::-webkit-scrollbar-thumb { background: #D5DBE5; border-radius: 8px; border: 2px solid var(--card); }
.slim::-webkit-scrollbar-thumb:hover { background: #B8C1CE; }
.slim { scrollbar-width: thin; scrollbar-color: #D5DBE5 var(--card); }

/* ---------- Decisions table ---------- */
.table-head { display: flex; align-items: center; justify-content: space-between;
  padding: 15px 20px; border-bottom: 1px solid var(--line); }
.table-title { display: inline-flex; align-items: center; gap: 8px; font-size: .66rem; font-weight: 700;
  letter-spacing: .11em; text-transform: uppercase; color: var(--text); }
.table-title .ic { color: var(--indigo); }
.badge-count { font-size: .63rem; color: var(--text2); padding: 4px 11px; border-radius: 999px;
  border: 1px solid var(--line); background: #F8FAFC; }
.table-wrap { max-height: 460px; overflow: auto; border-radius: 0 0 16px 16px; scroll-margin-top: 24px; }
table { width: 100%; border-collapse: collapse; }
th { position: sticky; top: 0; z-index: 2; padding: 10px 16px; text-align: left; font-size: .58rem;
  font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--text2);
  background: #F8FAFC; border-bottom: 1px solid var(--line); }
td { padding: 10px 16px; font-size: .73rem; border-bottom: 1px solid var(--line2); color: #344054; white-space: nowrap; }
tbody tr { transition: background .15s ease; }
tbody tr td:first-child { box-shadow: inset 0 0 0 0 var(--indigo); transition: box-shadow .18s ease; }
tbody tr:hover { background: #F8FAFF; }
tbody tr:hover td:first-child { box-shadow: inset 3px 0 0 0 var(--indigo); }
tbody tr:last-child td { border-bottom: none; }
.ta-r { text-align: right; }
.td-muted { color: var(--text2); }
.td-id { color: var(--indigoD); }
.td-empty { text-align: center; color: var(--text3); padding: 30px 0; font-size: .73rem; }
.type-pill { display: inline-block; padding: 3px 9px; border-radius: 6px; font-size: .6rem; font-weight: 700;
  letter-spacing: .04em; color: var(--indigoD); background: var(--indigoT); }
.badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .6rem; font-weight: 700; letter-spacing: .06em; }
.badge.BLOCK { color: var(--redD); background: var(--redT); }
.badge.REVIEW { color: var(--amberD); background: var(--amberT); }
.badge.APPROVE { color: var(--greenD); background: var(--greenT); }
.scorecell { display: inline-flex; flex-direction: column; gap: 4px; min-width: 78px; }
.scorebar { display: block; height: 4px; border-radius: 3px; background: var(--track); overflow: hidden; }
.scorebar i { display: block; height: 100%; border-radius: 3px; }

/* ---------- Footer ---------- */
.foot { display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap; padding: 2px 6px; }
.foot-note { font-size: .63rem; color: var(--text3); font-weight: 500; }
.foot-note b { color: var(--text2); }
.foot-right { display: flex; align-items: center; gap: 10px; }
.refbar { position: relative; width: 140px; height: 5px; border-radius: 99px; background: var(--track); overflow: hidden; }
#ref-fill { position: absolute; top: 0; left: 0; bottom: 0; width: 0; border-radius: 99px;
  background: linear-gradient(90deg, var(--indigo), var(--violet)); overflow: hidden; }
#ref-fill::after { content: ''; position: absolute; top: 0; bottom: 0; width: 40%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.6), transparent);
  animation: shimmer 1.6s linear infinite; }
@keyframes shimmer { from { transform: translateX(-110%); } to { transform: translateX(360%); } }
#ref-fill.run { animation: sweep 10s linear forwards; }
@keyframes sweep { from { width: 0; } to { width: 100%; } }

/* ---------- Responsive ---------- */
@media (max-width: 1100px) {
  .kpis { grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); }
  .grid-2, .grid-2b { grid-template-columns: 1fr; }
  .dist-layout { flex-direction: column; }
  .donut-wrap { margin: 0 auto; }
}
@media (max-width: 960px) {
  .sidebar { display: none; }
  .mobile-brand { display: flex; align-items: center; gap: 10px; }
  .mobile-brand span { font-family: var(--brand); font-size: 1rem; font-weight: 700; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
  html { scroll-behavior: auto; }
}
</style>
</head>
<body>

<div class="app">

  <!-- ======= Sidebar ======= -->
  <aside class="sidebar">
    <div class="sb-logo">
      <svg viewBox="0 0 24 24" width="32" height="32" aria-hidden="true">
        <defs>
          <linearGradient id="lg-logo" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stop-color="#5A5AE6"/><stop offset="1" stop-color="#12B76A"/>
          </linearGradient>
        </defs>
        <path d="M12 1.8l8.6 3.2v6.6c0 5.3-3.7 9-8.6 10.8-4.9-1.8-8.6-5.5-8.6-10.8V5L12 1.8z" fill="url(#lg-logo)"/>
        <path d="M8.3 12l2.7 2.7 5-5.6" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <div>
        <div class="sb-name">RazorSentry</div>
        <div class="sb-tag">Fraud Operations</div>
      </div>
    </div>
    <nav class="sb-nav">
      <a class="sb-link active" href="#top"><svg class="ic" viewBox="0 0 24 24"><rect x="3.5" y="3.5" width="7" height="7" rx="2"/><rect x="13.5" y="3.5" width="7" height="7" rx="2"/><rect x="3.5" y="13.5" width="7" height="7" rx="2"/><rect x="13.5" y="13.5" width="7" height="7" rx="2"/></svg>Overview</a>
      <a class="sb-link" href="#panel-spike"><svg class="ic" viewBox="0 0 24 24"><path d="M22 7l-8.5 8.5-5-5L2 17"/><path d="M16 7h6v6"/></svg>Spike Monitor</a>
      <a class="sb-link" href="#panel-drift"><svg class="ic" viewBox="0 0 24 24"><path d="M1 12q2.8-5 5.6 0t5.6 0 5.6 0 5.6 0"/><path d="M1 17q2.8-3.2 5.6 0t5.6 0 5.6 0 5.6 0"/></svg>Feature Drift</a>
      <a class="sb-link" href="#panel-psi"><svg class="ic" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5.2l3.4 2"/></svg>PSI History</a>
      <a class="sb-link" href="#panel-decisions"><svg class="ic" viewBox="0 0 24 24"><path d="M8.5 6h12M8.5 12h12M8.5 18h12M3.5 6h.01M3.5 12h.01M3.5 18h.01"/></svg>Decisions</a>
    </nav>
    <div class="sb-sys">
      <div class="sb-sys-title">System</div>
      <div class="sys-row"><span class="sys-k">Model</span><span class="sys-v mono" id="sys-model">—</span></div>
      <div class="sys-row"><span class="sys-k">Threshold θ</span><span class="sys-v mono" id="sys-threshold">—</span></div>
      <div class="sys-row"><span class="sys-k">Database</span><span class="sys-v" style="display:inline-flex;align-items:center;gap:6px"><span class="sys-dot" id="sys-db-dot"></span><span id="sys-db">—</span></span></div>
    </div>
  </aside>

  <!-- ======= Main ======= -->
  <main class="main" id="top">

    <div class="pagehead">
      <div>
        <div class="mobile-brand">
          <svg viewBox="0 0 24 24" width="26" height="26"><path d="M12 1.8l8.6 3.2v6.6c0 5.3-3.7 9-8.6 10.8-4.9-1.8-8.6-5.5-8.6-10.8V5L12 1.8z" fill="url(#lg-logo)"/><path d="M8.3 12l2.7 2.7 5-5.6" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>
          <span>RazorSentry</span>
        </div>
        <div class="ph-title">Operations Overview</div>
        <div class="ph-sub">
          <span class="livedot" id="live-dot"></span>
          <span class="livetxt">LIVE</span>
          <span class="mono" id="last-updated">connecting…</span>
        </div>
      </div>
      <div class="ph-right">
        <span class="pill"><svg class="ic" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/></svg>Auto-refresh · <b class="mono" id="countdown">10</b>s</span>
      </div>
    </div>

    <!-- ======= KPI cards ======= -->
    <section class="kpis">
      <div class="card kpi">
        <div class="kpi-head">
          <span class="kpi-ic ti-indigo"><svg class="ic" viewBox="0 0 24 24"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg></span>
          <span id="delta-total"></span>
        </div>
        <div class="kpi-value v-navy" id="val-total">—</div>
        <div class="kpi-label">Total Today</div>
        <div class="kpi-foot"><svg class="spark kpi-spark" id="spark-total" viewBox="0 0 140 34" preserveAspectRatio="none"></svg></div>
      </div>
      <div class="card kpi">
        <div class="kpi-head">
          <span class="kpi-ic ti-red"><svg class="ic" viewBox="0 0 24 24"><path d="M12 21.5C7.5 19.9 4 16.4 4 11.5V5.5L12 2.5l8 3v6c0 4.9-3.5 8.4-8 10z"/><path d="M9.4 9.4l5.2 5.2M14.6 9.4l-5.2 5.2"/></svg></span>
          <span id="delta-blocked"></span>
        </div>
        <div class="kpi-value v-red" id="val-blocked">—</div>
        <div class="kpi-label">Blocked</div>
        <div class="kpi-foot"><svg class="spark kpi-spark" id="spark-blocked" viewBox="0 0 140 34" preserveAspectRatio="none"></svg></div>
      </div>
      <div class="card kpi">
        <div class="kpi-head">
          <span class="kpi-ic ti-amber"><svg class="ic" viewBox="0 0 24 24"><path d="M2 12s3.8-7 10-7 10 7 10 7-3.8 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg></span>
          <span id="delta-review"></span>
        </div>
        <div class="kpi-value v-amber" id="val-review">—</div>
        <div class="kpi-label">Review Queue</div>
        <div class="kpi-foot"><svg class="spark kpi-spark" id="spark-review" viewBox="0 0 140 34" preserveAspectRatio="none"></svg></div>
      </div>
      <div class="card kpi">
        <div class="kpi-head">
          <span class="kpi-ic ti-green"><svg class="ic" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M8.1 12.4l2.7 2.7 5.1-5.7"/></svg></span>
          <span id="delta-approved"></span>
        </div>
        <div class="kpi-value v-green" id="val-approved">—</div>
        <div class="kpi-label">Approved</div>
        <div class="kpi-foot"><svg class="spark kpi-spark" id="spark-approved" viewBox="0 0 140 34" preserveAspectRatio="none"></svg></div>
      </div>
      <div class="card kpi">
        <div class="kpi-head">
          <span class="kpi-ic ti-violet"><svg class="ic" viewBox="0 0 24 24"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg></span>
          <span id="delta-latency"></span>
        </div>
        <div class="kpi-value v-violet" id="val-latency">—</div>
        <div class="kpi-label">Avg Latency</div>
        <div class="kpi-foot"><svg class="spark kpi-spark" id="spark-latency" viewBox="0 0 140 34" preserveAspectRatio="none"></svg></div>
      </div>
    </section>

    <!-- ======= Distribution + Spike ======= -->
    <div class="grid-2">
      <div class="card panel" id="panel-dist">
        <div class="panel-head">
          <span class="panel-title"><svg class="ic" viewBox="0 0 24 24"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>Decision Distribution</span>
        </div>
        <div class="dist-layout">
          <div class="donut-wrap">
            <svg class="donut" id="donut" viewBox="0 0 140 140" aria-hidden="true"></svg>
            <div class="donut-center">
              <div class="dc-val" id="donut-total">—</div>
              <div class="dc-lab">Total</div>
            </div>
          </div>
          <div class="dist-side">
            <div class="dist-bar" id="dist-bar"></div>
            <div class="dist-legend">
              <span class="lg"><i class="lgd" style="background:#F04438"></i>Block <b class="mono" id="pct-block">—</b></span>
              <span class="lg"><i class="lgd" style="background:#F79009"></i>Review <b class="mono" id="pct-review">—</b></span>
              <span class="lg"><i class="lgd" style="background:#12B76A"></i>Approve <b class="mono" id="pct-approve">—</b></span>
            </div>
            <div class="dist-note" id="dist-note">Loading…</div>
          </div>
        </div>
      </div>

      <div class="card panel" id="panel-spike">
        <div class="panel-head">
          <span class="panel-title"><svg class="ic" viewBox="0 0 24 24"><path d="M22 7l-8.5 8.5-5-5L2 17"/><path d="M16 7h6v6"/></svg>Fraud Spike Monitor · EWMA</span>
          <span class="chip warn" id="spike-chip"><span class="cdot"></span>…</span>
        </div>
        <div class="spike-hero">
          <span class="ewma-val" id="spike-ewma">—</span>
          <span class="ewma-lab">EWMA fraud rate · last 100 txns</span>
        </div>
        <div class="meter">
          <div class="meter-track">
            <div class="meter-fill" id="spike-fill" style="width:0%"></div>
            <i class="mtick" style="left:50%"></i>
            <i class="mtick" style="right:0"></i>
          </div>
          <div class="meter-scale mono"><span>0%</span><span>2.5%</span><span>5%+</span></div>
        </div>
        <div class="panel-sub" id="spike-detail">Loading…</div>
      </div>
    </div>

    <!-- ======= Drift + PSI History (scrollable) ======= -->
    <div class="grid-2b">
      <div class="card panel" id="panel-drift">
        <div class="panel-head">
          <span class="panel-title"><svg class="ic" viewBox="0 0 24 24"><path d="M1 12q2.8-5 5.6 0t5.6 0 5.6 0 5.6 0"/><path d="M1 17q2.8-3.2 5.6 0t5.6 0 5.6 0 5.6 0"/></svg>Feature Drift · PSI</span>
          <span class="chip warn" id="drift-chip"><span class="cdot"></span>…</span>
        </div>
        <div class="drift-meta mono" id="drift-detail">Loading drift telemetry…</div>
        <div id="psi-table"></div>
        <div class="psi-note">Log-scaled bars · warn ≥ 0.10 · alert ≥ 0.20 (tick marks)</div>
      </div>

      <div class="card panel" id="panel-psi">
        <div class="panel-head">
          <span class="panel-title"><svg class="ic" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5.2l3.4 2"/></svg>PSI History</span>
          <span class="panel-aux mono" id="psi-count">—</span>
        </div>
        <div class="psi-scroll slim" id="psi-timeline">
          <div class="empty">Waiting for drift checks…</div>
        </div>
      </div>
    </div>

    <!-- ======= Recent decisions ======= -->
    <section class="card" id="panel-decisions">
      <div class="table-head">
        <span class="table-title"><svg class="ic" viewBox="0 0 24 24"><path d="M8.5 6h12M8.5 12h12M8.5 18h12M3.5 6h.01M3.5 12h.01M3.5 18h.01"/></svg>Recent Decisions</span>
        <span class="badge-count mono" id="total-log-count">—</span>
      </div>
      <div class="table-wrap slim" id="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Transaction</th>
              <th>Type</th>
              <th class="ta-r">Amount</th>
              <th>Risk Score</th>
              <th>Decision</th>
              <th class="ta-r">Latency</th>
            </tr>
          </thead>
          <tbody id="decisions-body">
            <tr><td colspan="7" class="td-empty">Loading decisions…</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <footer class="foot">
      <span class="foot-note">RazorSentry · Fraud Operations Console · panels refresh automatically every 10s</span>
      <div class="foot-right">
        <span class="foot-note">next refresh in <b class="mono" id="countdown2">10</b>s</span>
        <span class="refbar"><i id="ref-fill"></i></span>
      </div>
    </footer>

  </main>
</div>

<script>
const $ = (id) => document.getElementById(id);
function fmtInt(v) { return Math.round(v).toLocaleString('en-US'); }
function num(v, d) { return (v === undefined || v === null || v === '') ? '—' : Number(v).toFixed(d); }

/* dir: +1 increase is good, -1 increase is bad, 0 neutral */
const METRICS = [
  { key: 'total',    dir: 0,  color: '#5A5AE6', fmt: fmtInt },
  { key: 'blocked',  dir: -1, color: '#F04438', fmt: fmtInt },
  { key: 'review',   dir: -1, color: '#F79009', fmt: fmtInt },
  { key: 'approved', dir: 1,  color: '#12B76A', fmt: fmtInt },
  { key: 'latency',  dir: -1, color: '#8B5CF6', fmt: (v) => v.toFixed(2) + 'ms' }
];
const hist = { total: [], blocked: [], review: [], approved: [], latency: [] };
let prev = null;

/* count-up animation */
function tween(el, from, to, dur, fmt) {
  if (!isFinite(to)) { el.textContent = '—'; return; }
  if (from === null || !isFinite(from) || from === to) { el.textContent = fmt(to); return; }
  const t0 = performance.now();
  const step = (t) => {
    const p = Math.min((t - t0) / dur, 1);
    const e = 1 - Math.pow(1 - p, 3);
    el.textContent = fmt(from + (to - from) * e);
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* delta pill */
function deltaChip(diff, dir, dec) {
  if (diff === null) return '';
  if (Math.abs(diff) < 0.005) return '<span class="dp flat">0.0%</span>';
  const up = diff > 0;
  const good = dir === 0 ? null : (up ? dir > 0 : dir < 0);
  const cls = dir === 0 ? 'flat' : (good ? 'up' : 'down');
  const arrow = up ? '+' : '–';
  const mag = dec ? Math.abs(diff).toFixed(1) : fmtInt(Math.abs(diff));
  return '<span class="dp ' + cls + '">' + arrow + mag + '</span>';
}

/* sparkline — only re-render + animate when data actually changed */
function setSpark(key, vals, color) {
  const svg = $('spark-' + key);
  const sig = vals.join(',');
  if (svg.dataset.sig === sig) return;
  svg.dataset.sig = sig;
  const w = 140, h = 34, p = 3;
  let inner;
  if (!vals || vals.length < 2) {
    inner = '<line x1="4" y1="' + (h - 6) + '" x2="' + (w - 4) + '" y2="' + (h - 6) + '" stroke="#E0E5EE" stroke-width="1.5" stroke-dasharray="3 4"/>';
  } else {
    const min = Math.min(...vals), max = Math.max(...vals);
    const span = (max - min) || max || 1;
    const pts = vals.map((v, i) => [
      p + (i / (vals.length - 1)) * (w - 2 * p),
      h - p - ((v - min) / span) * (h - 2 * p)
    ]);
    const line = pts.map(pt => pt[0].toFixed(1) + ' ' + pt[1].toFixed(1)).join(' ');
    const area = p + ' ' + (h - p) + ' ' + line + ' ' + (w - p) + ' ' + (h - p);
    const gid = 'sg-' + key;
    const last = pts[pts.length - 1];
    inner = '<defs><linearGradient id="' + gid + '" x1="0" y1="0" x2="0" y2="1">' +
      '<stop offset="0" stop-color="' + color + '" stop-opacity=".18"/>' +
      '<stop offset="1" stop-color="' + color + '" stop-opacity="0"/></linearGradient></defs>' +
      '<polygon points="' + area + '" fill="url(#' + gid + ')"/>' +
      '<polyline points="' + line + '" fill="none" stroke="' + color + '" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round"/>' +
      '<circle cx="' + last[0].toFixed(1) + '" cy="' + last[1].toFixed(1) + '" r="2.4" fill="' + color + '"/>';
  }
  svg.innerHTML = inner;
  /* line-draw animation */
  const pl = svg.querySelector('polyline');
  if (pl) {
    const len = pl.getTotalLength();
    pl.style.strokeDasharray = len;
    pl.style.strokeDashoffset = len;
    requestAnimationFrame(() => {
      pl.style.transition = 'stroke-dashoffset .7s cubic-bezier(.4,0,.2,1)';
      pl.style.strokeDashoffset = '0';
    });
    const dot = svg.querySelector('circle');
    if (dot) {
      dot.style.opacity = '0';
      setTimeout(() => { dot.style.transition = 'opacity .3s'; dot.style.opacity = '1'; }, 450);
    }
  }
}

/* donut */
function renderDonut(b, r, a) {
  const total = b + r + a;
  const defs = '<defs>' +
    '<linearGradient id="dgb" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#F97066"/><stop offset="1" stop-color="#F04438"/></linearGradient>' +
    '<linearGradient id="dgr" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#FDB022"/><stop offset="1" stop-color="#F79009"/></linearGradient>' +
    '<linearGradient id="dga" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3CC97C"/><stop offset="1" stop-color="#12B76A"/></linearGradient>' +
  '</defs>';
  let segs = '', off = 0;
  if (total > 0) {
    [['b', b], ['r', r], ['a', a]].forEach(seg => {
      if (seg[1] <= 0) return;
      const len = (seg[1] / total) * 100;
      segs += '<circle class="dn-seg" cx="70" cy="70" r="54" pathLength="100" stroke="url(#dg' + seg[0] + ')" ' +
        'stroke-dasharray="' + len.toFixed(2) + ' ' + (100 - len).toFixed(2) + '" stroke-dashoffset="' + (-off).toFixed(2) + '"/>';
      off += len;
    });
  }
  $('donut').innerHTML = defs + '<g transform="rotate(-90 70 70)"><circle class="dn-track" cx="70" cy="70" r="54" pathLength="100"/>' + segs + '</g>';
  $('donut-total').textContent = fmtInt(total);
}

function renderDist(b, r, a) {
  const total = b + r + a;
  if (total <= 0) return;
  $('dist-bar').innerHTML =
    '<div class="dist-block" style="width:' + (b / total * 100) + '%"></div>' +
    '<div class="dist-review" style="width:' + (r / total * 100) + '%"></div>' +
    '<div class="dist-approve" style="width:' + (a / total * 100) + '%"></div>';
  $('pct-block').textContent = (b / total * 100).toFixed(1) + '%';
  $('pct-review').textContent = (r / total * 100).toFixed(1) + '%';
  $('pct-approve').textContent = (a / total * 100).toFixed(1) + '%';
  $('dist-note').textContent = fmtInt(b) + ' blocked · ' + fmtInt(r) + ' review · ' + fmtInt(a) + ' approved today';
}

function setChip(id, kind, text) {
  const el = $(id);
  el.className = 'chip ' + kind;
  el.innerHTML = '<span class="cdot"></span>' + text;
}

/* PSI helpers (log scale, full scale = 5.0) */
const LOG6 = Math.log10(6);
function psiWidth(psi) {
  const p = Math.max(Number(psi) || 0, 0);
  return Math.min(100, Math.log10(1 + p) / LOG6 * 100);
}
const TICK_WARN = psiWidth(0.10).toFixed(1) + '%';
const TICK_ALERT = psiWidth(0.20).toFixed(1) + '%';

function renderPsiBars(fp, alerts) {
  if (!fp || !Object.keys(fp).length) return '<div class="empty">No feature-level PSI data</div>';
  const ticks = '<i class="pb-tick" style="left:' + TICK_WARN + '"></i><i class="pb-tick" style="left:' + TICK_ALERT + '"></i>';
  return Object.entries(fp).sort((x, y) => y[1] - x[1]).map(e => {
    const feat = e[0];
    const psi = Number(e[1]) || 0;
    const isAlert = alerts && alerts.indexOf(feat) !== -1;
    const isWarn = !isAlert && psi > 0.1;
    const col = isAlert ? '#F04438' : (isWarn ? '#F79009' : '#12B76A');
    const colD = isAlert ? '#B42318' : (isWarn ? '#B54708' : '#079451');
    const w = psiWidth(psi).toFixed(1);
    const mark = isAlert ? ' ⚠' : (isWarn ? ' ▲' : '');
    return '<div class="pb-row">' +
      '<span class="pb-name mono" title="' + feat + '">' + feat + '</span>' +
      '<span class="pb-track"><span class="pb-fill" style="width:' + w + '%;background:' + col + '"></span>' + ticks + '</span>' +
      '<span class="pb-val mono" style="color:' + colD + '">' + psi.toFixed(4) + mark + '</span>' +
    '</div>';
  }).join('');
}

/* PSI history — scrollable, newest first, scroll position preserved */
function renderTimeline(list) {
  const el = $('psi-timeline');
  if (!list || !list.length) {
    el.innerHTML = '<div class="empty">No drift checks recorded yet</div>';
    $('psi-count').textContent = '0 checks';
    return;
  }
  $('psi-count').textContent = list.length + (list.length === 1 ? ' check' : ' checks');
  const sig = list.map(h => h.timestamp + ':' + h.max_psi + ':' + (h.alert ? 1 : 0) + (h.warn ? 1 : 0)).join('|');
  if (el.dataset.sig === sig) return;
  el.dataset.sig = sig;
  const keep = el.scrollTop;
  el.innerHTML = list.slice().reverse().map(h => {
    const col = h.alert ? '#F04438' : (h.warn ? '#F79009' : '#12B76A');
    const colD = h.alert ? '#B42318' : (h.warn ? '#B54708' : '#079451');
    const cls = h.alert ? 'alert' : (h.warn ? 'warn' : 'ok');
    const w = psiWidth(h.max_psi).toFixed(1);
    return '<div class="tl-row">' +
      '<span class="tl-time mono">' + (h.timestamp || '—') + '</span>' +
      '<span class="tl-dot ' + cls + '"></span>' +
      '<span class="tl-track"><span class="tl-fill" style="width:' + w + '%;background:' + col + '"></span></span>' +
      '<span class="tl-psi mono" style="color:' + colD + '">' + Number(h.max_psi || 0).toFixed(3) + '</span>' +
      '<span class="tl-n mono">n=' + (h.samples !== undefined ? h.samples : '—') + '</span>' +
    '</div>';
  }).join('');
  el.scrollTop = keep;
}

/* decisions table */
function renderTable(rows) {
  const tbody = $('decisions-body');
  const wrap = $('table-wrap');
  if (!rows || !rows.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="td-empty">No decisions yet — score a transaction to begin</td></tr>';
    return;
  }
  const sig = rows.map(d => (d.timestamp || '') + (d.transaction_id || '') + (d.score || '') + (d.decision || '')).join('|');
  if (tbody.dataset.sig === sig) return;
  tbody.dataset.sig = sig;
  const keep = wrap.scrollTop;
  tbody.innerHTML = rows.map(d => {
    const ts = d.timestamp ? new Date(d.timestamp).toLocaleTimeString() : '—';
    const tid = d.transaction_id || '—';
    const amt = (d.amount !== undefined && d.amount !== null) ? '₹' + Number(d.amount).toLocaleString('en-IN', { maximumFractionDigits: 0 }) : '—';
    const lat = (d.latency_ms !== undefined && d.latency_ms !== null) ? Number(d.latency_ms).toFixed(1) + 'ms' : '—';
    const dec = d.decision || '—';
    const type = d.transaction_type || '—';
    let scoreHtml = '<span class="td-muted">—</span>';
    if (d.score !== undefined && d.score !== null) {
      const sc = Number(d.score);
      const col = sc >= 0.7 ? '#F04438' : (sc >= 0.4 ? '#F79009' : '#12B76A');
      const colD = sc >= 0.7 ? '#B42318' : (sc >= 0.4 ? '#B54708' : '#079451');
      scoreHtml = '<span class="scorecell"><span class="mono" style="color:' + colD + '">' + sc.toFixed(4) + '</span>' +
        '<span class="scorebar"><i style="width:' + (sc * 100).toFixed(1) + '%;background:' + col + '"></i></span></span>';
    }
    return '<tr>' +
      '<td class="mono td-muted">' + ts + '</td>' +
      '<td class="mono td-id" title="' + tid + '">' + tid.substring(0, 10) + '…</td>' +
      '<td><span class="type-pill">' + type + '</span></td>' +
      '<td class="mono ta-r">' + amt + '</td>' +
      '<td>' + scoreHtml + '</td>' +
      '<td><span class="badge ' + dec + '">' + dec + '</span></td>' +
      '<td class="mono ta-r td-muted">' + lat + '</td>' +
    '</tr>';
  }).join('');
  wrap.scrollTop = keep;
}

/* refresh countdown */
let cd = 10;
const cdEl = $('countdown'), cdEl2 = $('countdown2');
setInterval(() => { cd = Math.max(cd - 1, 0); cdEl.textContent = cd; cdEl2.textContent = cd; }, 1000);
function resetCountdown() {
  cd = 10; cdEl.textContent = '10'; cdEl2.textContent = '10';
  const f = $('ref-fill');
  f.classList.remove('run'); void f.offsetWidth; f.classList.add('run');
}

/* main refresh */
async function refresh() {
  try {
    const res = await fetch('/dashboard/stats');
    const data = await res.json();
    const s = data.stats;

    const cur = {
      total: Number(s.total_today) || 0,
      blocked: Number(s.blocked) || 0,
      review: Number(s.review) || 0,
      approved: Number(s.approved) || 0,
      latency: Number(s.avg_latency_ms) || 0
    };

    METRICS.forEach(m => {
      const from = prev ? prev[m.key] : null;
      const vEl = $('val-' + m.key);
      tween(vEl, from, cur[m.key], prev ? 500 : 800, m.fmt);
      /* flash the value when it changes */
      if (prev && cur[m.key] !== prev[m.key]) {
        vEl.classList.remove('flash'); void vEl.offsetWidth; vEl.classList.add('flash');
      }
      /* delta pill — only re-render (pop) when it changes */
      const dEl = $('delta-' + m.key);
      const dHtml = deltaChip(prev ? cur[m.key] - prev[m.key] : null, m.dir, m.key === 'latency');
      if (dEl.dataset.h !== dHtml) { dEl.dataset.h = dHtml; dEl.innerHTML = dHtml; }
      /* sparkline */
      const arr = hist[m.key];
      arr.push(cur[m.key]);
      if (arr.length > 40) arr.shift();
      setSpark(m.key, arr, m.color);
    });

    /* sidebar system card */
    $('sys-model').textContent = data.model_version || '—';
    $('sys-threshold').textContent = Number(data.operating_threshold).toFixed(2);
    $('sys-db').textContent = data.db_healthy ? 'Connected' : 'Unreachable';
    $('sys-db-dot').className = 'sys-dot' + (data.db_healthy ? '' : ' bad');
    $('live-dot').classList.remove('bad');
    $('last-updated').textContent = 'updated ' + new Date().toLocaleTimeString();
    $('total-log-count').textContent = (s.total_in_log !== undefined ? fmtInt(s.total_in_log) : '—') + ' in audit log';

    renderDist(cur.blocked, cur.review, cur.approved);
    renderDonut(cur.blocked, cur.review, cur.approved);

    const spike = data.spike_alert || {};
    const rate = Number(spike.ewma_rate) || 0;
    $('spike-ewma').textContent = (rate * 100).toFixed(2) + '%';
    const fillEl = $('spike-fill');
    fillEl.style.width = (Math.min(rate / 0.05, 1) * 100).toFixed(1) + '%';
    if (spike.spike_detected) {
      setChip('spike-chip', 'danger', 'Spike detected');
      fillEl.style.background = '#F04438';
      $('spike-detail').textContent = 'EWMA fraud rate has breached the alert threshold — investigate immediately.';
    } else {
      setChip('spike-chip', 'ok', 'Normal');
      fillEl.style.background = rate < 0.025 ? '#12B76A' : (rate < 0.05 ? '#F79009' : '#F04438');
      $('spike-detail').textContent = 'Rolling EWMA of block-rate across the last 100 scored transactions · meter scale 0–5%.';
    }

    const drift = data.drift_alert || {};
    const psiTable = $('psi-table');
    if (!drift.drift_checked) {
      setChip('drift-chip', 'warn', 'Standby');
      $('drift-detail').textContent = drift.reason || 'Insufficient data for drift check';
      psiTable.innerHTML = '<div class="empty">Not enough samples yet</div>';
    } else if (drift.alert) {
      setChip('drift-chip', 'danger', 'Drift alert');
      $('drift-detail').textContent = 'max PSI ' + num(drift.max_psi, 4) + ' · ' + (drift.samples_checked || '—') + ' samples · retraining recommended';
      psiTable.innerHTML = renderPsiBars(drift.feature_psi, drift.alert_features);
    } else if (drift.warn) {
      setChip('drift-chip', 'warn', 'Moderate');
      $('drift-detail').textContent = 'max PSI ' + num(drift.max_psi, 4) + ' · ' + (drift.samples_checked || '—') + ' samples · monitor closely';
      psiTable.innerHTML = renderPsiBars(drift.feature_psi, drift.alert_features);
    } else {
      setChip('drift-chip', 'ok', 'Stable');
      $('drift-detail').textContent = 'max PSI ' + num(drift.max_psi, 4) + ' · ' + (drift.samples_checked || '—') + ' samples · distributions match training';
      psiTable.innerHTML = renderPsiBars(drift.feature_psi, []);
    }

    renderTimeline(data.drift_history || []);
    renderTable(data.last_10_decisions);

    prev = cur;
  } catch (e) {
    $('last-updated').textContent = 'connection lost — retrying';
    $('live-dot').classList.add('bad');
    setChip('spike-chip', 'warn', 'Offline');
    setChip('drift-chip', 'warn', 'Offline');
  }
  resetCountdown();
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>"""
    return HTMLResponse(content=html)