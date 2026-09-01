# Builds the stats payload that the dashboard HTML fetches every 10 seconds

from datetime import datetime, timezone
from src.audit import get_recent_decisions, check_db_health


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
    scores = [d["score"] for d in recent if "score" in d]
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
        "scores": scores,
    }
