# Fraud spike detection uses EWMA — deliberately LLM-free for speed and auditability

import numpy as np

BASELINE_FRAUD_RATE = 0.02
SPIKE_MULTIPLIER = 2.0
EWMA_SPAN = 20


# Computes EWMA-smoothed flagged rate and returns a spike alert dict
def check_fraud_spike(recent_scores: list[float], window: int = 100, threshold: float = 0.5) -> dict:
    if not recent_scores:
        return {"spike_detected": False, "ewma_rate": 0.0, "message": "No data"}

    scores = np.array(recent_scores[-window:], dtype=float)
    flagged = (scores >= threshold).astype(float)

    alpha = 2.0 / (EWMA_SPAN + 1)
    ewma = float(flagged[0])
    for val in flagged[1:]:
        ewma = alpha * val + (1 - alpha) * ewma

    spike = bool(ewma > BASELINE_FRAUD_RATE * SPIKE_MULTIPLIER)

    result: dict = {"spike_detected": spike, "ewma_rate": round(float(ewma), 6)}
    if spike:
        result["message"] = (
            f"Fraud spike detected — EWMA rate {ewma:.2%} exceeds "
            f"{SPIKE_MULTIPLIER}x baseline ({BASELINE_FRAUD_RATE:.2%})"
        )
    return result
