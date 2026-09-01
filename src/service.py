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
<title>RazorSentry — Fraud Operations Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; padding: 24px; }
  h1 { font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
  .subtitle { font-size: 0.8rem; color: #64748b; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: #1e2433; border-radius: 10px; padding: 20px; border: 1px solid #2d3748; }
  .card .label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
  .card .value { font-size: 2rem; font-weight: 700; }
  .card .value.block { color: #f87171; }
  .card .value.review { color: #fbbf24; }
  .card .value.approve { color: #34d399; }
  .card .value.total { color: #60a5fa; }
  .card .value.latency { color: #a78bfa; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
  .three-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 24px; }
  .panel { background: #1e2433; border-radius: 10px; border: 1px solid #2d3748; padding: 16px; }
  .panel-title { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px; }
  .alert-status { font-size: 0.95rem; font-weight: 600; }
  .alert-ok { color: #34d399; }
  .alert-warn { color: #fbbf24; }
  .alert-danger { color: #f87171; }
  .dist-bar { display: flex; height: 12px; border-radius: 6px; overflow: hidden; margin: 10px 0 6px; }
  .dist-block { background: #f87171; }
  .dist-review { background: #fbbf24; }
  .dist-approve { background: #34d399; }
  .dist-legend { display: flex; gap: 12px; font-size: 0.7rem; color: #94a3b8; }
  .dist-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 3px; }
  .table-section { background: #1e2433; border-radius: 10px; border: 1px solid #2d3748; overflow: hidden; margin-bottom: 24px; }
  .table-header { padding: 14px 20px; font-size: 0.85rem; font-weight: 600; border-bottom: 1px solid #2d3748; color: #94a3b8; display: flex; justify-content: space-between; align-items: center; }
  .badge-count { font-size: 0.7rem; background: #2d3748; padding: 2px 8px; border-radius: 10px; }
  table { width: 100%; border-collapse: collapse; }
  th { padding: 10px 16px; text-align: left; font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; border-bottom: 1px solid #2d3748; }
  td { padding: 9px 16px; font-size: 0.78rem; border-bottom: 1px solid #1a2035; font-family: 'SF Mono', 'Fira Code', monospace; }
  tr:last-child td { border-bottom: none; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }
  .badge.BLOCK { background: #7f1d1d; color: #f87171; }
  .badge.REVIEW { background: #78350f; color: #fbbf24; }
  .badge.APPROVE { background: #064e3b; color: #34d399; }
  .footer { font-size: 0.72rem; color: #475569; text-align: center; margin-top: 8px; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #34d399; margin-right: 6px; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  .timeline-row { display: flex; align-items: center; gap: 8px; padding: 4px 0; border-bottom: 1px solid #1a2035; font-size: 0.72rem; }
  .timeline-row:last-child { border-bottom: none; }
  .psi-bar-wrap { flex: 1; background: #0f1117; border-radius: 2px; height: 6px; }
  .psi-bar-fill { height: 6px; border-radius: 2px; }
</style>
</head>
<body>
<h1>🛡️ RazorSentry</h1>
<p class="subtitle"><span class="dot"></span>Fraud Operations Dashboard &nbsp;|&nbsp; <span id="last-updated">Loading...</span></p>

<div class="grid">
  <div class="card"><div class="label">Total Today</div><div class="value total" id="total">—</div></div>
  <div class="card"><div class="label">Blocked</div><div class="value block" id="blocked">—</div></div>
  <div class="card"><div class="label">Review</div><div class="value review" id="review">—</div></div>
  <div class="card"><div class="label">Approved</div><div class="value approve" id="approved">—</div></div>
  <div class="card"><div class="label">Avg Latency</div><div class="value latency" id="latency">—</div></div>
</div>

<div class="two-col">
  <div class="panel">
    <div class="panel-title">📊 Decision Distribution</div>
    <div class="dist-bar" id="dist-bar"></div>
    <div class="dist-legend">
      <span><span class="dist-dot" style="background:#f87171"></span>Block <span id="pct-block">—</span></span>
      <span><span class="dist-dot" style="background:#fbbf24"></span>Review <span id="pct-review">—</span></span>
      <span><span class="dist-dot" style="background:#34d399"></span>Approve <span id="pct-approve">—</span></span>
    </div>
    <div style="font-size:0.7rem;color:#475569;margin-top:8px" id="dist-note"></div>
  </div>
  <div class="panel">
    <div class="panel-title">⚡ Fraud Spike Monitor (EWMA)</div>
    <div class="alert-status" id="spike-status">Loading...</div>
    <div style="font-size:0.72rem;color:#64748b;margin-top:6px" id="spike-detail"></div>
  </div>
</div>

<div class="two-col">
  <div class="panel">
    <div class="panel-title">📉 Feature Drift Monitor (PSI)</div>
    <div class="alert-status" id="drift-status">Loading...</div>
    <div style="font-size:0.72rem;color:#64748b;margin-top:4px" id="drift-detail"></div>
    <div id="psi-table" style="margin-top:10px;font-size:0.7rem;font-family:monospace"></div>
  </div>
  <div class="panel">
    <div class="panel-title">⏱️ PSI History (last 20 checks)</div>
    <div id="psi-timeline" style="margin-top:4px">
      <div style="color:#475569;font-size:0.72rem">Waiting for drift checks...</div>
    </div>
  </div>
</div>

<div class="table-section">
  <div class="table-header">
    <span>Last 20 Decisions</span>
    <span class="badge-count" id="total-log-count"></span>
  </div>
  <table>
    <thead>
      <tr>
        <th>Time</th>
        <th>Transaction ID</th>
        <th>Type</th>
        <th>Amount</th>
        <th>Score</th>
        <th>Decision</th>
        <th>Latency</th>
      </tr>
    </thead>
    <tbody id="decisions-body">
      <tr><td colspan="7" style="color:#475569;text-align:center;padding:24px">Loading...</td></tr>
    </tbody>
  </table>
</div>

<div class="footer">
  Model: <span id="model-ver">—</span> &nbsp;|&nbsp;
  Threshold: <span id="threshold">—</span> &nbsp;|&nbsp;
  DB: <span id="db-health">—</span> &nbsp;|&nbsp;
  Auto-refresh every 10s
</div>

<script>
function renderPsiTable(featurePsi, alertFeatures) {
  if (!featurePsi || Object.keys(featurePsi).length === 0) return '';
  const rows = Object.entries(featurePsi)
    .sort((a, b) => b[1] - a[1])
    .map(([feat, psi]) => {
      const isAlert = alertFeatures && alertFeatures.includes(feat);
      const isWarn = psi > 0.1 && !isAlert;
      const barWidth = Math.min(psi * 20, 100).toFixed(0);
      const color = isAlert ? '#f87171' : isWarn ? '#fbbf24' : '#34d399';
      const flag = isAlert ? ' ⚠' : isWarn ? ' ↑' : '';
      return `<div style="margin-bottom:5px;display:flex;align-items:center;gap:6px">
        <span style="width:110px;color:#94a3b8;font-size:0.68rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${feat}">${feat}</span>
        <div style="flex:1;background:#0f1117;border-radius:2px;height:6px">
          <div style="width:${barWidth}%;background:${color};height:6px;border-radius:2px"></div>
        </div>
        <span style="width:60px;text-align:right;color:${color};font-size:0.68rem">${psi.toFixed(4)}${flag}</span>
      </div>`;
    }).join('');
  return `<div style="margin-top:4px">${rows}</div>`;
}

function renderPsiTimeline(history) {
  if (!history || history.length === 0) {
    return '<div style="color:#475569;font-size:0.72rem">No drift checks recorded yet</div>';
  }
  return history.slice().reverse().map(h => {
    const color = h.alert ? '#f87171' : h.warn ? '#fbbf24' : '#34d399';
    const label = h.alert ? '🔴' : h.warn ? '🟡' : '✅';
    const barW = Math.min(h.max_psi * 15, 100).toFixed(0);
    return `<div class="timeline-row">
      <span style="color:#475569;width:52px">${h.timestamp}</span>
      <span>${label}</span>
      <div class="psi-bar-wrap">
        <div class="psi-bar-fill" style="width:${barW}%;background:${color}"></div>
      </div>
      <span style="color:${color};width:52px;text-align:right">${h.max_psi.toFixed(3)}</span>
      <span style="color:#475569;width:40px;font-size:0.65rem">n=${h.samples}</span>
    </div>`;
  }).join('');
}

async function refresh() {
  try {
    const res = await fetch('/dashboard/stats');
    const data = await res.json();
    const s = data.stats;

    document.getElementById('total').textContent = s.total_today;
    document.getElementById('blocked').textContent = s.blocked;
    document.getElementById('review').textContent = s.review;
    document.getElementById('approved').textContent = s.approved;
    document.getElementById('latency').textContent = s.avg_latency_ms + 'ms';
    document.getElementById('model-ver').textContent = data.model_version;
    document.getElementById('threshold').textContent = parseFloat(data.operating_threshold).toFixed(2);
    document.getElementById('db-health').textContent = data.db_healthy ? '✅ Connected' : '❌ Unreachable';
    document.getElementById('last-updated').textContent = 'Updated ' + new Date().toLocaleTimeString();
    document.getElementById('total-log-count').textContent = s.total_in_log + ' total in log';

    const total = s.blocked + s.review + s.approved;
    if (total > 0) {
      const pBlock = (s.blocked / total * 100).toFixed(1);
      const pReview = (s.review / total * 100).toFixed(1);
      const pApprove = (s.approved / total * 100).toFixed(1);
      document.getElementById('dist-bar').innerHTML =
        `<div class="dist-block" style="width:${pBlock}%"></div>` +
        `<div class="dist-review" style="width:${pReview}%"></div>` +
        `<div class="dist-approve" style="width:${pApprove}%"></div>`;
      document.getElementById('pct-block').textContent = pBlock + '%';
      document.getElementById('pct-review').textContent = pReview + '%';
      document.getElementById('pct-approve').textContent = pApprove + '%';
      document.getElementById('dist-note').textContent =
        `${s.blocked} blocked · ${s.review} review · ${s.approved} approved out of ${total} today`;
    }

    const spike = data.spike_alert;
    const spikeEl = document.getElementById('spike-status');
    if (spike.spike_detected) {
      spikeEl.textContent = '🔴 SPIKE DETECTED';
      spikeEl.className = 'alert-status alert-danger';
      document.getElementById('spike-detail').textContent = 'EWMA rate: ' + (spike.ewma_rate * 100).toFixed(2) + '%';
    } else {
      spikeEl.textContent = '✅ Normal';
      spikeEl.className = 'alert-status alert-ok';
      document.getElementById('spike-detail').textContent = 'EWMA rate: ' + (spike.ewma_rate * 100).toFixed(2) + '%';
    }

    const drift = data.drift_alert;
    const driftEl = document.getElementById('drift-status');
    const driftDetail = document.getElementById('drift-detail');
    const psiTable = document.getElementById('psi-table');
    if (!drift.drift_checked) {
      driftEl.textContent = '⏳ Insufficient data';
      driftEl.className = 'alert-status alert-warn';
      driftDetail.textContent = drift.reason || '';
      psiTable.innerHTML = '';
    } else if (drift.alert) {
      driftEl.textContent = '🔴 DRIFT ALERT — Retrain recommended';
      driftEl.className = 'alert-status alert-danger';
      driftDetail.textContent = 'Max PSI: ' + drift.max_psi + ' | Samples: ' + (drift.samples_checked || '—');
      psiTable.innerHTML = renderPsiTable(drift.feature_psi, drift.alert_features);
    } else if (drift.warn) {
      driftEl.textContent = '🟡 Moderate drift detected';
      driftEl.className = 'alert-status alert-warn';
      driftDetail.textContent = 'Max PSI: ' + drift.max_psi + ' | Samples: ' + (drift.samples_checked || '—');
      psiTable.innerHTML = renderPsiTable(drift.feature_psi, drift.alert_features);
    } else {
      driftEl.textContent = '✅ Stable — distribution matches training';
      driftEl.className = 'alert-status alert-ok';
      driftDetail.textContent = 'Max PSI: ' + (drift.max_psi || 0) + ' | Samples: ' + (drift.samples_checked || '—');
      psiTable.innerHTML = renderPsiTable(drift.feature_psi, []);
    }

    document.getElementById('psi-timeline').innerHTML = renderPsiTimeline(data.drift_history || []);

    const tbody = document.getElementById('decisions-body');
    const rows = data.last_10_decisions;
    if (!rows || rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" style="color:#475569;text-align:center;padding:24px">No decisions yet</td></tr>';
      return;
    }
    tbody.innerHTML = rows.map(d => {
      const ts = d.timestamp ? new Date(d.timestamp).toLocaleTimeString() : '—';
      const tid = d.transaction_id || '—';
      const amt = d.amount ? '₹' + Number(d.amount).toLocaleString('en-IN', {maximumFractionDigits: 0}) : '—';
      const score = d.score !== undefined ? d.score.toFixed(4) : '—';
      const lat = d.latency_ms ? d.latency_ms.toFixed(1) + 'ms' : '—';
      const dec = d.decision || '—';
      const type = d.transaction_type || '—';
      return '<tr>' +
        '<td>' + ts + '</td>' +
        '<td>' + tid.substring(0, 12) + '...</td>' +
        '<td>' + type + '</td>' +
        '<td>' + amt + '</td>' +
        '<td>' + score + '</td>' +
        '<td><span class="badge ' + dec + '">' + dec + '</span></td>' +
        '<td>' + lat + '</td>' +
        '</tr>';
    }).join('');
  } catch(e) {
    document.getElementById('last-updated').textContent = 'Error fetching data';
  }
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
