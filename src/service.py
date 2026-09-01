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
from src.audit import check_db_health, get_decision, get_recent_decisions, init_db, log_decision
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
<meta name="theme-color" content="#05070F">
<title>RazorSentry — Fraud Operations Dashboard</title>
<style>
:root {
  --bg: #05070F;
  --text: #E7EDF8;
  --muted: #8B95AD;
  --faint: #5A6478;
  --mono: 'SF Mono','Cascadia Code','Fira Code','JetBrains Mono',Consolas,monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: var(--bg); color: var(--text);
  min-height: 100vh; padding: 22px clamp(14px, 3vw, 40px) 30px;
  -webkit-font-smoothing: antialiased; overflow-x: hidden;
}
::selection { background: rgba(129,140,248,.4); }
.mono { font-family: var(--mono); font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }

/* ---------- Aurora background ---------- */
.aurora { position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none; }
.blob { position: absolute; border-radius: 50%; filter: blur(110px); opacity: .5; will-change: transform; }
.b1 { width: 46vw; height: 46vw; min-width: 520px; min-height: 520px;
  background: radial-gradient(circle at 35% 35%, #7C3AED 0%, rgba(124,58,237,0) 62%);
  top: -18vh; left: -10vw; animation: d1 26s ease-in-out infinite alternate; }
.b2 { width: 40vw; height: 40vw; min-width: 460px; min-height: 460px;
  background: radial-gradient(circle at 60% 40%, #0EA5E9 0%, rgba(14,165,233,0) 60%);
  top: 8vh; right: -12vw; animation: d2 21s ease-in-out infinite alternate; }
.b3 { width: 52vw; height: 52vw; min-width: 600px; min-height: 600px;
  background: radial-gradient(circle at 50% 50%, #4F46E5 0%, rgba(79,70,229,0) 64%);
  bottom: -24vh; left: 22vw; animation: d3 31s ease-in-out infinite alternate; }
.b4 { width: 26vw; height: 26vw; min-width: 300px; min-height: 300px; opacity: .32;
  background: radial-gradient(circle at 50% 50%, #D946EF 0%, rgba(217,70,239,0) 60%);
  top: 55vh; left: -8vw; animation: d4 24s ease-in-out infinite alternate; }
@keyframes d1 { to { transform: translate3d(9vw, 7vh, 0) scale(1.18); } }
@keyframes d2 { to { transform: translate3d(-8vw, 10vh, 0) scale(1.12); } }
@keyframes d3 { to { transform: translate3d(-10vw, -8vh, 0) scale(1.08); } }
@keyframes d4 { to { transform: translate3d(10vw, -10vh, 0) scale(1.25); } }
.grid-bg { position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: linear-gradient(rgba(139,149,173,.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(139,149,173,.05) 1px, transparent 1px);
  background-size: 46px 46px;
  -webkit-mask-image: radial-gradient(ellipse 85% 60% at 50% 8%, #000 25%, transparent 78%);
  mask-image: radial-gradient(ellipse 85% 60% at 50% 8%, #000 25%, transparent 78%); }

/* ---------- Glass ---------- */
.shell { position: relative; z-index: 1; max-width: 1240px; margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }
.glass {
  position: relative;
  background: linear-gradient(165deg, rgba(255,255,255,.06), rgba(255,255,255,.015) 55%, rgba(129,140,248,.03));
  border: 1px solid rgba(255,255,255,.09);
  border-radius: 18px;
  -webkit-backdrop-filter: blur(18px) saturate(150%);
  backdrop-filter: blur(18px) saturate(150%);
  box-shadow: 0 24px 60px -20px rgba(2,6,18,.75), inset 0 1px 0 rgba(255,255,255,.07);
}
.glass::before { content: ''; position: absolute; top: -1px; left: 10%; right: 10%; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(167,139,250,.55), transparent); pointer-events: none; }

/* ---------- Icons ---------- */
.ic { width: 15px; height: 15px; stroke: currentColor; fill: none; stroke-width: 1.8;
  stroke-linecap: round; stroke-linejoin: round; flex: none; }

/* ---------- Header ---------- */
.top { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 16px 22px; flex-wrap: wrap; }
.brand { display: flex; align-items: center; gap: 14px; }
.brand > svg { filter: drop-shadow(0 4px 18px rgba(129,140,248,.5)); }
h1 { font-size: 1.55rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.1;
  background: linear-gradient(100deg, #E0F2FE 0%, #67E8F9 28%, #A78BFA 62%, #F0ABFC 100%);
  background-size: 220% 100%;
  -webkit-background-clip: text; background-clip: text; color: transparent; -webkit-text-fill-color: transparent;
  filter: drop-shadow(0 2px 16px rgba(103,232,249,.25));
  animation: sheen 9s linear infinite; }
@keyframes sheen { to { background-position: 220% 0; } }
.subtitle { display: flex; align-items: center; gap: 8px; font-size: .74rem; color: var(--muted); margin-top: 4px; flex-wrap: wrap; }
.live-dot { width: 7px; height: 7px; border-radius: 50%; flex: none;
  background: linear-gradient(135deg,#34D399,#22D3FE); box-shadow: 0 0 10px rgba(52,211,153,.9);
  animation: pulse 2.2s ease-in-out infinite; }
.live-dot.bad { background: #FB7185; box-shadow: 0 0 10px rgba(251,113,133,.9); }
.live-txt { color: #6EE7B7; font-weight: 700; letter-spacing: .14em; font-size: .62rem; }
@keyframes pulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: .55; transform: scale(.85); } }
.hchips { display: flex; gap: 8px; flex-wrap: wrap; }
.f-chip { display: inline-flex; align-items: center; gap: 7px; font-size: .68rem; color: var(--muted);
  padding: 6px 11px; border-radius: 999px; border: 1px solid rgba(255,255,255,.09); background: rgba(255,255,255,.04); }
.f-chip .ic { width: 13px; height: 13px; color: #818CF8; }
.f-chip b { color: #CBD5E1; font-weight: 600; }
.dot-sm { width: 7px; height: 7px; border-radius: 50%; background: #34D399; box-shadow: 0 0 8px rgba(52,211,153,.8); }
.dot-sm.bad { background: #FB7185; box-shadow: 0 0 8px rgba(251,113,133,.8); }

/* ---------- Gradient text ---------- */
.grad-txt { -webkit-background-clip: text; background-clip: text; color: transparent; -webkit-text-fill-color: transparent; }
.g-cyan   { background: linear-gradient(135deg,#A5F3FC,#22D3EE 60%,#3B82F6); filter: drop-shadow(0 3px 16px rgba(34,211,238,.35)); }
.g-red    { background: linear-gradient(135deg,#FECDD3,#FB7185 55%,#E11D48); filter: drop-shadow(0 3px 16px rgba(244,63,94,.35)); }
.g-amber  { background: linear-gradient(135deg,#FDE68A,#FBBF24 55%,#D97706); filter: drop-shadow(0 3px 16px rgba(245,158,11,.3)); }
.g-green  { background: linear-gradient(135deg,#A7F3D0,#34D399 55%,#059669); filter: drop-shadow(0 3px 16px rgba(16,185,129,.35)); }
.g-violet { background: linear-gradient(135deg,#DDD6FE,#A78BFA 55%,#7C3AED); filter: drop-shadow(0 3px 16px rgba(167,139,250,.4)); }

/* ---------- KPI cards ---------- */
.kpis { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; }
.kpi { padding: 16px 18px 12px; display: flex; flex-direction: column; gap: 8px; overflow: hidden;
  transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease; }
.kpi:hover { transform: translateY(-3px); border-color: rgba(129,140,248,.35);
  box-shadow: 0 30px 70px -22px rgba(2,6,18,.85), 0 0 30px -6px rgba(129,140,248,.25), inset 0 1px 0 rgba(255,255,255,.08); }
.kpi-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.kpi-label { display: inline-flex; align-items: center; gap: 7px; font-size: .62rem; font-weight: 600;
  letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
.kpi-ic { display: inline-flex; color: var(--muted); }
.kpi-ic .ic { width: 13px; height: 13px; }
.kpi-value { font-size: 2.05rem; font-weight: 800; letter-spacing: -.02em; line-height: 1; font-variant-numeric: tabular-nums; }
.kpi-spark { margin-top: auto; }
.spark { width: 100%; height: 34px; display: block; overflow: visible; }
.kpi.flash { animation: kpiFlash .8s ease; }
@keyframes kpiFlash {
  0% { box-shadow: 0 0 0 1px rgba(129,140,248,.55), 0 0 34px -4px rgba(129,140,248,.45), inset 0 1px 0 rgba(255,255,255,.07); }
  100% { box-shadow: 0 24px 60px -20px rgba(2,6,18,.75), inset 0 1px 0 rgba(255,255,255,.07); }
}
.d-chip { font-size: .6rem; font-weight: 700; padding: 3px 8px; border-radius: 999px;
  font-variant-numeric: tabular-nums; white-space: nowrap; }
.d-pos  { color: #6EE7B7; background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.3); }
.d-neg  { color: #FDA4AF; background: rgba(244,63,94,.12); border: 1px solid rgba(244,63,94,.3); }
.d-neu  { color: #67E8F9; background: rgba(34,211,238,.1); border: 1px solid rgba(34,211,238,.28); }
.d-zero { color: var(--faint); background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08); }

/* ---------- Panels ---------- */
.grid-2  { display: grid; grid-template-columns: 1.15fr 1fr; gap: 14px; }
.grid-2b { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.panel { padding: 18px 20px; }
.panel-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 14px; }
.panel-title { display: inline-flex; align-items: center; gap: 8px; font-size: .66rem; font-weight: 700;
  letter-spacing: .13em; text-transform: uppercase; color: #B6C2D9; }
.panel-title .ic { color: #818CF8; }
.panel-aux { font-size: .64rem; color: var(--faint); }
.panel-sub { font-size: .7rem; color: var(--muted); line-height: 1.55; }
.empty { padding: 20px 0; text-align: center; color: var(--faint); font-size: .7rem; }

/* status chips */
.chip { display: inline-flex; align-items: center; gap: 6px; font-size: .64rem; font-weight: 700;
  letter-spacing: .1em; padding: 5px 11px; border-radius: 999px; text-transform: uppercase; }
.chip .cdot { width: 6px; height: 6px; border-radius: 50%; }
.chip.ok { color: #6EE7B7; background: rgba(16,185,129,.1); border: 1px solid rgba(16,185,129,.35); box-shadow: 0 0 16px rgba(16,185,129,.12); }
.chip.ok .cdot { background: #34D399; box-shadow: 0 0 8px rgba(52,211,153,.8); }
.chip.warn { color: #FCD34D; background: rgba(245,158,11,.1); border: 1px solid rgba(245,158,11,.35); box-shadow: 0 0 16px rgba(245,158,11,.12); }
.chip.warn .cdot { background: #FBBF24; box-shadow: 0 0 8px rgba(251,191,36,.8); }
.chip.danger { color: #FDA4AF; background: rgba(244,63,94,.12); border: 1px solid rgba(244,63,94,.45);
  box-shadow: 0 0 20px rgba(244,63,94,.2); animation: chipPulse 1.6s ease-in-out infinite; }
.chip.danger .cdot { background: #FB7185; box-shadow: 0 0 8px rgba(251,113,133,.9); }
@keyframes chipPulse { 0%,100% { box-shadow: 0 0 0 0 rgba(244,63,94,.28); } 50% { box-shadow: 0 0 0 7px rgba(244,63,94,0); } }

/* ---------- Distribution ---------- */
.dist-layout { display: flex; align-items: center; gap: 22px; }
.donut-wrap { position: relative; width: 148px; height: 148px; flex: none; }
#donut { width: 148px; height: 148px; display: block; filter: drop-shadow(0 6px 18px rgba(3,6,18,.6)); }
.dn-track { fill: none; stroke: rgba(255,255,255,.06); stroke-width: 13; }
.dn-seg { fill: none; stroke-width: 13; }
.dn-block   { stroke: url(#dg-block);   filter: drop-shadow(0 0 6px rgba(244,63,94,.4)); }
.dn-review  { stroke: url(#dg-review);  filter: drop-shadow(0 0 6px rgba(245,158,11,.35)); }
.dn-approve { stroke: url(#dg-approve); filter: drop-shadow(0 0 6px rgba(16,185,129,.35)); }
.donut-center { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 2px; }
.dc-val { font-size: 1.55rem; font-weight: 800; }
.dc-lab { font-size: .58rem; letter-spacing: .16em; color: var(--faint); text-transform: uppercase; }
.dist-side { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 10px; }
.dist-bar { display: flex; height: 12px; border-radius: 7px; overflow: hidden;
  background: rgba(255,255,255,.05); box-shadow: inset 0 1px 3px rgba(0,0,0,.45); }
.dist-block   { background: linear-gradient(180deg,#FB7185,#E11D48); box-shadow: 0 0 10px rgba(244,63,94,.4); }
.dist-review  { background: linear-gradient(180deg,#FCD34D,#F59E0B); box-shadow: 0 0 10px rgba(245,158,11,.35); }
.dist-approve { background: linear-gradient(180deg,#6EE7B7,#059669); box-shadow: 0 0 10px rgba(16,185,129,.35); }
.dist-legend { display: flex; gap: 16px; flex-wrap: wrap; }
.lg-item { display: inline-flex; align-items: center; gap: 6px; font-size: .7rem; color: var(--muted); }
.lg-dot { width: 8px; height: 8px; border-radius: 3px; background: var(--c); box-shadow: 0 0 10px var(--c); }
.lg-val { font-weight: 700; color: #DDE5F2; }
.dist-note { font-size: .68rem; color: var(--faint); }

/* ---------- Spike monitor ---------- */
.spike-hero { display: flex; align-items: baseline; gap: 10px; margin-bottom: 12px; }
.ewma-val { font-size: 1.9rem; font-weight: 800; letter-spacing: -.02em; }
.ewma-lab { font-size: .66rem; color: var(--muted); }
.meter { margin: 4px 0 12px; }
.meter-track { position: relative; height: 10px; border-radius: 6px; background: rgba(255,255,255,.06);
  overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,.45); }
.meter-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg,#34D399,#22D3EE);
  box-shadow: 0 0 14px rgba(34,211,238,.5); transition: width .7s cubic-bezier(.22,1,.36,1), background .4s; }
.meter-fill.danger { background: linear-gradient(90deg,#FB7185,#E11D48); box-shadow: 0 0 16px rgba(244,63,94,.6);
  animation: meterPulse 1.2s ease-in-out infinite; }
@keyframes meterPulse { 0%,100% { opacity: 1; } 50% { opacity: .6; } }
.meter-scale { display: flex; justify-content: space-between; font-size: .6rem; color: var(--faint); margin-top: 5px; }

/* ---------- Drift bars ---------- */
.drift-meta { font-size: .68rem; color: var(--muted); margin-bottom: 12px; }
.pb-row { display: flex; align-items: center; gap: 10px; padding: 5px 0; }
.pb-name { width: 128px; font-size: .66rem; color: #A9B4C9; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: none; }
.pb-track { display: block; position: relative; flex: 1; height: 7px; border-radius: 4px; background: rgba(255,255,255,.05); }
.pb-fill { display: block; height: 100%; border-radius: 4px; transition: width .6s ease; }
.pb-tick { position: absolute; top: -3px; bottom: -3px; width: 1px; background: rgba(255,255,255,.22); }
.pb-val { width: 70px; text-align: right; font-size: .66rem; flex: none; font-weight: 600; }
.psi-note { margin-top: 10px; font-size: .62rem; color: var(--faint); }

/* ---------- PSI history (scrollable) ---------- */
.psi-scroll { max-height: 335px; overflow-y: auto; overflow-x: hidden; padding-right: 8px;
  -webkit-mask-image: linear-gradient(180deg, transparent 0, #000 14px, #000 calc(100% - 14px), transparent 100%);
  mask-image: linear-gradient(180deg, transparent 0, #000 14px, #000 calc(100% - 14px), transparent 100%); }
.tl-row { display: flex; align-items: center; gap: 8px; padding: 6px 2px;
  border-bottom: 1px solid rgba(255,255,255,.05); border-radius: 6px; transition: background .15s; }
.tl-row:hover { background: rgba(129,140,248,.07); }
.tl-row:last-child { border-bottom: none; }
.tl-time { width: 56px; font-size: .64rem; color: var(--muted); flex: none; }
.tl-dot { width: 7px; height: 7px; border-radius: 50%; flex: none; }
.tl-dot.ok { background: #34D399; box-shadow: 0 0 8px rgba(52,211,153,.75); }
.tl-dot.warn { background: #FBBF24; box-shadow: 0 0 8px rgba(251,191,36,.75); }
.tl-dot.alert { background: #FB7185; box-shadow: 0 0 8px rgba(251,113,133,.8); animation: pulse 1.6s infinite; }
.tl-track { display: block; flex: 1; height: 6px; border-radius: 3px; background: rgba(255,255,255,.05); overflow: hidden; }
.tl-fill { display: block; height: 100%; border-radius: 3px; }
.tl-psi { width: 52px; text-align: right; font-size: .66rem; font-weight: 600; flex: none; }
.tl-n { width: 52px; text-align: right; font-size: .6rem; color: var(--faint); flex: none; }

/* custom scrollbars */
.slim-scroll { scrollbar-width: thin; scrollbar-color: rgba(129,140,248,.6) rgba(255,255,255,.05); }
.slim-scroll::-webkit-scrollbar { width: 6px; height: 6px; }
.slim-scroll::-webkit-scrollbar-track { background: rgba(255,255,255,.04); border-radius: 99px; }
.slim-scroll::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#818CF8,#22D3EE); border-radius: 99px; }

/* ---------- Decisions table ---------- */
.table-card { overflow: hidden; }
.table-head { display: flex; align-items: center; justify-content: space-between;
  padding: 15px 20px; border-bottom: 1px solid rgba(255,255,255,.07); }
.table-title { display: inline-flex; align-items: center; gap: 8px; font-size: .66rem; font-weight: 700;
  letter-spacing: .13em; text-transform: uppercase; color: #B6C2D9; }
.table-title .ic { color: #818CF8; }
.badge-count { font-size: .64rem; color: var(--muted); padding: 4px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,.09); background: rgba(255,255,255,.04); }
.table-wrap { max-height: 470px; overflow: auto; }
table { width: 100%; border-collapse: collapse; }
th { position: sticky; top: 0; z-index: 2; padding: 10px 16px; text-align: left; font-size: .6rem;
  font-weight: 700; letter-spacing: .11em; text-transform: uppercase; color: var(--muted);
  background: rgba(9,13,26,.97); border-bottom: 1px solid rgba(255,255,255,.08); }
td { padding: 10px 16px; font-size: .74rem; border-bottom: 1px solid rgba(255,255,255,.045); color: #C9D3E4; }
tbody tr { transition: background .16s ease; }
tbody tr:hover { background: linear-gradient(90deg, rgba(124,58,237,.12), rgba(34,211,238,.05) 70%); }
tbody tr:last-child td { border-bottom: none; }
.ta-r, th.ta-r { text-align: right; }
.td-muted { color: var(--muted); }
.td-id { color: #A5B4FC; }
.td-empty { text-align: center; color: var(--faint); padding: 30px 0; font-size: .74rem; }
.type-pill { display: inline-block; padding: 3px 9px; border-radius: 6px; font-size: .62rem; font-weight: 600;
  letter-spacing: .05em; color: #93C5FD; background: rgba(59,130,246,.1); border: 1px solid rgba(59,130,246,.25); }
.badge { display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 999px;
  font-size: .62rem; font-weight: 700; letter-spacing: .08em; }
.badge::before { content: ''; width: 5px; height: 5px; border-radius: 50%; background: currentColor; box-shadow: 0 0 7px currentColor; }
.badge.BLOCK { color: #FDA4AF; background: rgba(244,63,94,.13); border: 1px solid rgba(244,63,94,.4); box-shadow: 0 0 14px rgba(244,63,94,.15); }
.badge.REVIEW { color: #FCD34D; background: rgba(245,158,11,.13); border: 1px solid rgba(245,158,11,.4); box-shadow: 0 0 14px rgba(245,158,11,.15); }
.badge.APPROVE { color: #6EE7B7; background: rgba(16,185,129,.13); border: 1px solid rgba(16,185,129,.4); box-shadow: 0 0 14px rgba(16,185,129,.15); }
.score-cell { display: inline-flex; flex-direction: column; gap: 4px; min-width: 70px; }
.score-track { display: block; height: 3px; border-radius: 2px; background: rgba(255,255,255,.08); overflow: hidden; }
.score-fill { display: block; height: 100%; border-radius: 2px; }

/* ---------- Footer ---------- */
.foot { display: flex; align-items: center; justify-content: space-between; gap: 14px; padding: 13px 20px; flex-wrap: wrap; }
.foot-note { font-size: .66rem; color: var(--faint); }
.foot-refresh { display: flex; align-items: center; gap: 10px; }
.refresh-label { font-size: .66rem; color: var(--muted); }
.refresh-label b { color: #CBD5E1; }
.refresh-bar { position: relative; width: 150px; height: 4px; border-radius: 99px; background: rgba(255,255,255,.08); overflow: hidden; }
#refresh-fill { position: absolute; top: 0; left: 0; bottom: 0; width: 0; border-radius: 99px;
  background: linear-gradient(90deg,#22D3EE,#818CF8,#A855F7); box-shadow: 0 0 10px rgba(129,140,248,.5); }
#refresh-fill.run { animation: sweep 10s linear forwards; }
@keyframes sweep { from { width: 0; } to { width: 100%; } }

/* ---------- Responsive ---------- */
@media (max-width: 1100px) {
  .kpis { grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
  .grid-2, .grid-2b { grid-template-columns: 1fr; }
  .dist-layout { flex-direction: column; }
  .donut-wrap { margin: 0 auto; }
}
@media (max-width: 560px) {
  body { padding: 12px; }
  h1 { font-size: 1.2rem; }
  .top { padding: 12px 14px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
</head>
<body>

<div class="aurora" aria-hidden="true">
  <div class="blob b1"></div><div class="blob b2"></div>
  <div class="blob b3"></div><div class="blob b4"></div>
</div>
<div class="grid-bg" aria-hidden="true"></div>

<div class="shell">

  <!-- ======= Header ======= -->
  <header class="top glass">
    <div class="brand">
      <svg viewBox="0 0 24 24" width="38" height="38" aria-hidden="true">
        <defs>
          <linearGradient id="lg-logo" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stop-color="#22D3EE"/><stop offset=".55" stop-color="#818CF8"/><stop offset="1" stop-color="#A855F7"/>
          </linearGradient>
        </defs>
        <path d="M12 1.8l8.6 3.2v6.6c0 5.3-3.7 9-8.6 10.8-4.9-1.8-8.6-5.5-8.6-10.8V5L12 1.8z" fill="url(#lg-logo)"/>
        <path d="M8.3 12l2.7 2.7 5-5.6" fill="none" stroke="#070A16" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <div>
        <h1>RazorSentry</h1>
        <div class="subtitle">
          <span class="live-dot" id="live-dot"></span>
          <span class="live-txt">LIVE</span>
          <span>Fraud Operations Dashboard</span>
          <span class="mono" id="last-updated">connecting…</span>
        </div>
      </div>
    </div>
    <div class="hchips">
      <span class="f-chip"><svg class="ic" viewBox="0 0 24 24"><rect x="6.5" y="6.5" width="11" height="11" rx="2"/><path d="M9.5 2.5v3M14.5 2.5v3M9.5 18.5v3M14.5 18.5v3M2.5 9.5h3M2.5 14.5h3M18.5 9.5h3M18.5 14.5h3"/></svg><b class="mono" id="model-ver">—</b></span>
      <span class="f-chip"><svg class="ic" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.2"/></svg>θ <b class="mono" id="threshold">—</b></span>
      <span class="f-chip"><span class="dot-sm" id="db-dot"></span>DB <b id="db-health">—</b></span>
    </div>
  </header>

  <!-- ======= KPI cards ======= -->
  <section class="kpis">
    <div class="glass kpi" id="card-total">
      <div class="kpi-top">
        <span class="kpi-label"><span class="kpi-ic" style="color:#22D3EE"><svg class="ic" viewBox="0 0 24 24"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg></span>Total Today</span>
        <span id="delta-total"></span>
      </div>
      <div class="kpi-value grad-txt g-cyan" id="val-total">—</div>
      <div class="kpi-spark" id="spark-total"></div>
    </div>
    <div class="glass kpi" id="card-blocked">
      <div class="kpi-top">
        <span class="kpi-label"><span class="kpi-ic" style="color:#FB7185"><svg class="ic" viewBox="0 0 24 24"><path d="M12 21.5C7.5 19.9 4 16.4 4 11.5V5.5L12 2.5l8 3v6c0 4.9-3.5 8.4-8 10z"/><path d="M9.4 9.4l5.2 5.2M14.6 9.4l-5.2 5.2"/></svg></span>Blocked</span>
        <span id="delta-blocked"></span>
      </div>
      <div class="kpi-value grad-txt g-red" id="val-blocked">—</div>
      <div class="kpi-spark" id="spark-blocked"></div>
    </div>
    <div class="glass kpi" id="card-review">
      <div class="kpi-top">
        <span class="kpi-label"><span class="kpi-ic" style="color:#FBBF24"><svg class="ic" viewBox="0 0 24 24"><path d="M2 12s3.8-6.8 10-6.8S22 12 22 12s-3.8 6.8-10 6.8S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg></span>Review Queue</span>
        <span id="delta-review"></span>
      </div>
      <div class="kpi-value grad-txt g-amber" id="val-review">—</div>
      <div class="kpi-spark" id="spark-review"></div>
    </div>
    <div class="glass kpi" id="card-approved">
      <div class="kpi-top">
        <span class="kpi-label"><span class="kpi-ic" style="color:#34D399"><svg class="ic" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9.2"/><path d="M8.1 12.4l2.7 2.7 5.1-5.7"/></svg></span>Approved</span>
        <span id="delta-approved"></span>
      </div>
      <div class="kpi-value grad-txt g-green" id="val-approved">—</div>
      <div class="kpi-spark" id="spark-approved"></div>
    </div>
    <div class="glass kpi" id="card-latency">
      <div class="kpi-top">
        <span class="kpi-label"><span class="kpi-ic" style="color:#A78BFA"><svg class="ic" viewBox="0 0 24 24"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg></span>Avg Latency</span>
        <span id="delta-latency"></span>
      </div>
      <div class="kpi-value grad-txt g-violet" id="val-latency">—</div>
      <div class="kpi-spark" id="spark-latency"></div>
    </div>
  </section>

  <!-- ======= Distribution + Spike ======= -->
  <div class="grid-2">
    <div class="panel glass">
      <div class="panel-head">
        <span class="panel-title"><svg class="ic" viewBox="0 0 24 24"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>Decision Distribution</span>
      </div>
      <div class="dist-layout">
        <div class="donut-wrap">
          <svg id="donut" viewBox="0 0 140 140"><circle class="dn-track" cx="70" cy="70" r="55" pathLength="100"/></svg>
          <div class="donut-center">
            <div class="dc-val grad-txt g-cyan" id="donut-total">—</div>
            <div class="dc-lab">total</div>
          </div>
        </div>
        <div class="dist-side">
          <div class="dist-bar" id="dist-bar"></div>
          <div class="dist-legend">
            <span class="lg-item"><span class="lg-dot" style="--c:#F43F5E"></span>Block <b class="lg-val mono" id="pct-block">—</b></span>
            <span class="lg-item"><span class="lg-dot" style="--c:#F59E0B"></span>Review <b class="lg-val mono" id="pct-review">—</b></span>
            <span class="lg-item"><span class="lg-dot" style="--c:#059669"></span>Approve <b class="lg-val mono" id="pct-approve">—</b></span>
          </div>
          <div class="dist-note" id="dist-note">Loading…</div>
        </div>
      </div>
    </div>

    <div class="panel glass">
      <div class="panel-head">
        <span class="panel-title"><svg class="ic" viewBox="0 0 24 24"><path d="M22 7l-8.5 8.5-5-5L2 17"/><path d="M16 7h6v6"/></svg>Fraud Spike Monitor · EWMA</span>
        <span class="chip warn" id="spike-chip"><span class="cdot"></span>…</span>
      </div>
      <div class="spike-hero">
        <span class="ewma-val grad-txt g-cyan" id="spike-ewma">—</span>
        <span class="ewma-lab">EWMA fraud rate</span>
      </div>
      <div class="meter">
        <div class="meter-track"><div class="meter-fill" id="spike-fill" style="width:0%"></div></div>
        <div class="meter-scale"><span>0%</span><span>2.5%</span><span>5%+</span></div>
      </div>
      <div class="panel-sub" id="spike-detail">Loading…</div>
    </div>
  </div>

  <!-- ======= Drift + PSI History (scrollable) ======= -->
  <div class="grid-2b">
    <div class="panel glass">
      <div class="panel-head">
        <span class="panel-title"><svg class="ic" viewBox="0 0 24 24"><path d="M1 12q2.8-5 5.6 0t5.6 0t5.6 0t5.6 0"/><path d="M1 17q2.8-3.2 5.6 0t5.6 0t5.6 0t5.6 0"/></svg>Feature Drift · PSI</span>
        <span class="chip warn" id="drift-chip"><span class="cdot"></span>…</span>
      </div>
      <div class="drift-meta mono" id="drift-detail">Loading drift telemetry…</div>
      <div id="psi-table"></div>
      <div class="psi-note">Tick marks show PSI warn (0.10) and alert (0.20) thresholds · bars use a logarithmic scale</div>
    </div>

    <div class="panel glass">
      <div class="panel-head">
        <span class="panel-title"><svg class="ic" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9.2"/><path d="M12 7v5.2l3.4 2"/></svg>PSI History</span>
        <span class="panel-aux mono" id="psi-count">—</span>
      </div>
      <div class="psi-scroll slim-scroll" id="psi-timeline">
        <div class="empty">Waiting for drift checks…</div>
      </div>
    </div>
  </div>

  <!-- ======= Recent decisions ======= -->
  <section class="glass table-card">
    <div class="table-head">
      <span class="table-title"><svg class="ic" viewBox="0 0 24 24"><path d="M8.5 6h12M8.5 12h12M8.5 18h12M3.5 6h.01M3.5 12h.01M3.5 18h.01"/></svg>Recent Decisions</span>
      <span class="badge-count mono" id="total-log-count">—</span>
    </div>
    <div class="table-wrap slim-scroll" id="table-wrap">
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

  <!-- ======= Footer ======= -->
  <footer class="foot glass">
    <span class="foot-note">RazorSentry · real-time fraud decisioning · all panels refresh automatically</span>
    <div class="foot-refresh">
      <span class="refresh-label">next refresh in <b class="mono" id="countdown">10</b>s</span>
      <span class="refresh-bar"><i id="refresh-fill"></i></span>
    </div>
  </footer>

</div>

<script>
const $ = (id) => document.getElementById(id);

function fmtInt(v) { return Math.round(v).toLocaleString('en-US'); }
function num(v, d) { return (v === undefined || v === null || v === '') ? '—' : Number(v).toFixed(d); }

/* ---------- metric config (dir: +1 = increase is good, -1 = increase is bad, 0 = neutral) ---------- */
const METRICS = [
  { key: 'total',    dir: 0,  color: '#22D3EE', fmt: fmtInt },
  { key: 'blocked',  dir: -1, color: '#FB7185', fmt: fmtInt },
  { key: 'review',   dir: -1, color: '#FBBF24', fmt: fmtInt },
  { key: 'approved', dir: 1,  color: '#34D399', fmt: fmtInt },
  { key: 'latency',  dir: -1, color: '#A78BFA', fmt: (v) => v.toFixed(2) + 'ms' }
];
const hist = { total: [], blocked: [], review: [], approved: [], latency: [] };
let prev = null;

/* ---------- animated count-up ---------- */
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

/* ---------- delta chip vs last refresh ---------- */
function deltaChip(diff, dir, dec) {
  if (diff === null) return '';
  if (Math.abs(diff) < 0.005) return '<span class="d-chip d-zero">— 0</span>';
  const up = diff > 0;
  const good = dir === 0 ? null : (up ? dir > 0 : dir < 0);
  const cls = dir === 0 ? 'd-neu' : (good ? 'd-pos' : 'd-neg');
  const arrow = up ? '▲' : '▼';
  const mag = Math.abs(diff);
  const txt = dec ? mag.toFixed(1) : fmtInt(mag);
  return '<span class="d-chip ' + cls + '">' + arrow + ' ' + txt + '</span>';
}

/* ---------- sparkline ---------- */
function sparkSVG(vals, key, color) {
  const w = 140, h = 34, p = 3;
  if (!vals || vals.length < 2) {
    return '<svg class="spark" viewBox="0 0 ' + w + ' ' + h + '"><line x1="4" y1="' + (h - 6) + '" x2="' + (w - 4) + '" y2="' + (h - 6) + '" stroke="rgba(255,255,255,.12)" stroke-width="1" stroke-dasharray="3 4"/></svg>';
  }
  const min = Math.min(...vals), max = Math.max(...vals);
  const span = (max - min) || max || 1;
  const pts = vals.map((v, i) => [
    p + (i / (vals.length - 1)) * (w - 2 * p),
    h - p - ((v - min) / span) * (h - 2 * p)
  ]);
  const line = pts.map(pt => pt[0].toFixed(1) + ' ' + pt[1].toFixed(1)).join(' ');
  const area = (p + ' ' + (h - p) + ' ' + line + ' ' + (w - p) + ' ' + (h - p));
  const gid = 'sg-' + key;
  const last = pts[pts.length - 1];
  return '<svg class="spark" viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none">' +
    '<defs><linearGradient id="' + gid + '" x1="0" y1="0" x2="0" y2="1">' +
    '<stop offset="0" stop-color="' + color + '" stop-opacity=".3"/>' +
    '<stop offset="1" stop-color="' + color + '" stop-opacity="0"/></linearGradient></defs>' +
    '<polygon points="' + area + '" fill="url(#' + gid + ')"/>' +
    '<polyline points="' + line + '" fill="none" stroke="' + color + '" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>' +
    '<circle cx="' + last[0].toFixed(1) + '" cy="' + last[1].toFixed(1) + '" r="2.4" fill="' + color + '"/>' +
  '</svg>';
}

/* ---------- donut ---------- */
function renderDonut(b, r, a) {
  const total = b + r + a;
  const svg = $('donut');
  const defs = '<defs>' +
    '<linearGradient id="dg-block" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#FDA4AF"/><stop offset="1" stop-color="#E11D48"/></linearGradient>' +
    '<linearGradient id="dg-review" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#FDE68A"/><stop offset="1" stop-color="#F59E0B"/></linearGradient>' +
    '<linearGradient id="dg-approve" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6EE7B7"/><stop offset="1" stop-color="#059669"/></linearGradient>' +
  '</defs>';
  let segs = '', off = 0;
  if (total > 0) {
    [['block', b], ['review', r], ['approve', a]].forEach(seg => {
      if (seg[1] <= 0) return;
      const len = (seg[1] / total) * 100;
      segs += '<circle class="dn-seg dn-' + seg[0] + '" cx="70" cy="70" r="55" pathLength="100" ' +
        'stroke-dasharray="' + len.toFixed(2) + ' ' + (100 - len).toFixed(2) + '" stroke-dashoffset="' + (-off).toFixed(2) + '"/>';
      off += len;
    });
  }
  svg.innerHTML = defs + '<g transform="rotate(-90 70 70)"><circle class="dn-track" cx="70" cy="70" r="55" pathLength="100"/>' + segs + '</g>';
  $('donut-total').textContent = fmtInt(total);
}

function renderDist(b, r, a) {
  const total = b + r + a;
  if (total <= 0) return;
  const pB = b / total * 100, pR = r / total * 100, pA = a / total * 100;
  $('dist-bar').innerHTML =
    '<div class="dist-block" style="width:' + pB + '%"></div>' +
    '<div class="dist-review" style="width:' + pR + '%"></div>' +
    '<div class="dist-approve" style="width:' + pA + '%"></div>';
  $('pct-block').textContent = pB.toFixed(1) + '%';
  $('pct-review').textContent = pR.toFixed(1) + '%';
  $('pct-approve').textContent = pA.toFixed(1) + '%';
  $('dist-note').textContent = fmtInt(b) + ' blocked · ' + fmtInt(r) + ' review · ' + fmtInt(a) + ' approved today';
}

/* ---------- status chip ---------- */
function setChip(id, kind, text) {
  const el = $(id);
  el.className = 'chip ' + kind;
  el.innerHTML = '<span class="cdot"></span>' + text;
}

/* ---------- PSI helpers (log scale, full scale ≈ 5.0) ---------- */
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
    const col = isAlert ? '#FB7185' : (isWarn ? '#FBBF24' : '#34D399');
    const w = psiWidth(psi).toFixed(1);
    const mark = isAlert ? ' ⚠' : (isWarn ? ' ▲' : '');
    return '<div class="pb-row">' +
      '<span class="pb-name mono" title="' + feat + '">' + feat + '</span>' +
      '<span class="pb-track"><span class="pb-fill" style="width:' + w + '%;background:linear-gradient(90deg,' + col + '55,' + col + ');box-shadow:0 0 8px ' + col + '40"></span>' + ticks + '</span>' +
      '<span class="pb-val mono" style="color:' + col + '">' + psi.toFixed(4) + mark + '</span>' +
    '</div>';
  }).join('');
}

/* ---------- PSI history (scrollable, position preserved) ---------- */
function renderTimeline(list) {
  const el = $('psi-timeline');
  if (!list || !list.length) {
    el.innerHTML = '<div class="empty">No drift checks recorded yet</div>';
    $('psi-count').textContent = '0 checks';
    return;
  }
  $('psi-count').textContent = list.length + ' checks';
  const sig = list.map(h => h.timestamp + ':' + h.max_psi + ':' + (h.alert ? 1 : 0) + (h.warn ? 1 : 0)).join('|');
  if (el.dataset.sig === sig) return;
  el.dataset.sig = sig;
  const keep = el.scrollTop;
  el.innerHTML = list.slice().reverse().map(h => {
    const col = h.alert ? '#FB7185' : (h.warn ? '#FBBF24' : '#34D399');
    const cls = h.alert ? 'alert' : (h.warn ? 'warn' : 'ok');
    const w = psiWidth(h.max_psi).toFixed(1);
    return '<div class="tl-row">' +
      '<span class="tl-time mono">' + (h.timestamp || '—') + '</span>' +
      '<span class="tl-dot ' + cls + '"></span>' +
      '<span class="tl-track"><span class="tl-fill" style="width:' + w + '%;background:linear-gradient(90deg,' + col + '55,' + col + ')"></span></span>' +
      '<span class="tl-psi mono" style="color:' + col + '">' + Number(h.max_psi || 0).toFixed(3) + '</span>' +
      '<span class="tl-n mono">n=' + (h.samples !== undefined ? h.samples : '—') + '</span>' +
    '</div>';
  }).join('');
  el.scrollTop = keep;
}

/* ---------- decisions table ---------- */
function renderTable(rows) {
  const tbody = $('decisions-body');
  const wrap = $('table-wrap');
  if (!rows || !rows.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="td-empty">No decisions yet — score a transaction to begin</td></tr>';
    return;
  }
  const sig = rows.map(d => (d.timestamp || '') + (d.transaction_id || '') + (d.score || '')).join('|');
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
      const col = sc >= 0.7 ? '#FB7185' : (sc >= 0.4 ? '#FBBF24' : '#34D399');
      scoreHtml = '<span class="score-cell"><span class="mono" style="color:' + col + '">' + sc.toFixed(4) + '</span>' +
        '<span class="score-track"><span class="score-fill" style="width:' + (sc * 100).toFixed(1) + '%;background:linear-gradient(90deg,' + col + '66,' + col + ')"></span></span></span>';
    }
    return '<tr>' +
      '<td class="mono td-muted">' + ts + '</td>' +
      '<td class="mono td-id">' + tid.substring(0, 12) + '…</td>' +
      '<td><span class="type-pill">' + type + '</span></td>' +
      '<td class="mono ta-r">' + amt + '</td>' +
      '<td>' + scoreHtml + '</td>' +
      '<td><span class="badge ' + dec + '">' + dec + '</span></td>' +
      '<td class="mono ta-r td-muted">' + lat + '</td>' +
    '</tr>';
  }).join('');
  wrap.scrollTop = keep;
}

/* ---------- refresh countdown ---------- */
let cd = 10;
const cdEl = $('countdown');
setInterval(() => { cd = Math.max(cd - 1, 0); cdEl.textContent = cd; }, 1000);
function resetRefreshUI() {
  cd = 10; cdEl.textContent = '10';
  const fill = $('refresh-fill');
  fill.classList.remove('run'); void fill.offsetWidth; fill.classList.add('run');
}

/* ---------- main refresh ---------- */
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

    /* KPIs: count-up + delta + sparkline + change flash */
    METRICS.forEach(m => {
      const from = prev ? prev[m.key] : 0;
      tween($('val-' + m.key), from, cur[m.key], prev ? 450 : 900, m.fmt);
      $('delta-' + m.key).innerHTML = deltaChip(prev ? cur[m.key] - prev[m.key] : null, m.dir, m.key === 'latency');
      const arr = hist[m.key];
      arr.push(cur[m.key]);
      if (arr.length > 40) arr.shift();
      $('spark-' + m.key).innerHTML = sparkSVG(arr, m.key, m.color);
      if (prev && cur[m.key] !== prev[m.key]) {
        const card = $('card-' + m.key);
        card.classList.remove('flash'); void card.offsetWidth; card.classList.add('flash');
      }
    });

    /* header */
    $('model-ver').textContent = data.model_version || '—';
    $('threshold').textContent = Number(data.operating_threshold).toFixed(2);
    $('db-health').textContent = data.db_healthy ? 'Connected' : 'Unreachable';
    $('db-dot').className = 'dot-sm' + (data.db_healthy ? '' : ' bad');
    $('live-dot').classList.remove('bad');
    $('last-updated').textContent = '· updated ' + new Date().toLocaleTimeString();
    $('total-log-count').textContent = fmtInt(s.total_in_log) + ' in audit log';

    /* distribution */
    renderDist(cur.blocked, cur.review, cur.approved);
    renderDonut(cur.blocked, cur.review, cur.approved);

    /* spike monitor */
    const spike = data.spike_alert || {};
    const rate = Number(spike.ewma_rate) || 0;
    $('spike-ewma').textContent = (rate * 100).toFixed(2) + '%';
    $('spike-fill').style.width = (Math.min(rate / 0.05, 1) * 100).toFixed(1) + '%';
    if (spike.spike_detected) {
      setChip('spike-chip', 'danger', 'Spike detected');
      $('spike-fill').classList.add('danger');
      $('spike-detail').textContent = 'EWMA fraud rate has breached the alert threshold — investigate immediately.';
    } else {
      setChip('spike-chip', 'ok', 'Normal');
      $('spike-fill').classList.remove('danger');
      $('spike-detail').textContent = 'Rolling EWMA of block-rate across the last 100 scored transactions · meter scale 0–5%.';
    }

    /* drift monitor */
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

    /* psi history + decisions */
    renderTimeline(data.drift_history || []);
    renderTable(data.last_10_decisions);

    prev = cur;
    resetRefreshUI();
  } catch (e) {
    $('last-updated').textContent = '· connection lost — retrying';
    $('live-dot').classList.add('bad');
    setChip('spike-chip', 'warn', 'Offline');
    setChip('drift-chip', 'warn', 'Offline');
  }
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>"""
    return HTMLResponse(content=html)