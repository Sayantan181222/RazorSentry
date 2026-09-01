# Builds dashboard stats and maintains an in-memory drift history for timeline display

from datetime import datetime, timezone
from src.audit import get_recent_decisions, check_db_health

_drift_history: list[dict] = []
MAX_DRIFT_HISTORY = 20


# Returns aggregated decision stats for the dashboard from the last 200 decisions
def get_dashboard_stats(recent: list[dict]) -> dict:
    today = datetime.now(timezone.utc).date()
    today_decisions = [
        d for d in recent
        if d.get("timestamp") and d["timestamp"][:10] == str(today)
    ]
    total_today = len(today_decisions)
    blocked = sum(1 for d in today_decisions if d.get("decision") == "BLOCK")
    review = sum(1 for d in today_decisions if d.get("decision") == "REVIEW")
    approved = sum(1 for d in today_decisions if d.get("decision") == "APPROVE")
    avg_latency = (
        round(sum(d["latency_ms"] for d in recent if "latency_ms" in d) / len(recent), 2)
        if recent else 0
    )
    return {
        "total_today": total_today,
        "blocked": blocked,
        "review": review,
        "approved": approved,
        "avg_latency_ms": avg_latency,
        "total_in_log": len(recent),
    }


# Records a drift check result into the in-memory history list
def record_drift_history(drift_result: dict) -> None:
    global _drift_history
    if not drift_result.get("drift_checked"):
        return
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "max_psi": drift_result.get("max_psi", 0),
        "alert": drift_result.get("alert", False),
        "warn": drift_result.get("warn", False),
        "samples": drift_result.get("samples_checked", 0),
    }
    _drift_history.append(entry)
    if len(_drift_history) > MAX_DRIFT_HISTORY:
        _drift_history = _drift_history[-MAX_DRIFT_HISTORY:]


# Returns the current drift history list for the dashboard timeline
def get_drift_history() -> list[dict]:
    return _drift_history
