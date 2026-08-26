import pytest
from httpx import AsyncClient

FRAUD_LIKE_TX = {
    "transaction_id": "test_fraud_001",
    "step": 1,
    "type": "CASH_OUT",
    "amount": 490000.0,
    "nameOrig": "C1000000001",
    "oldbalanceOrg": 490000.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C9000000001",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 490000.0,
}

LEGIT_LIKE_TX = {
    "transaction_id": "test_legit_001",
    "step": 2,
    "type": "PAYMENT",
    "amount": 800.0,
    "nameOrig": "C1000000002",
    "oldbalanceOrg": 45000.0,
    "newbalanceOrig": 44200.0,
    "nameDest": "M9000000002",
    "oldbalanceDest": 5000.0,
    "newbalanceDest": 5800.0,
}

BATCH_TXS = [
    FRAUD_LIKE_TX,
    LEGIT_LIKE_TX,
    {
        "transaction_id": "test_transfer_001",
        "step": 3,
        "type": "TRANSFER",
        "amount": 200000.0,
        "nameOrig": "C1000000003",
        "oldbalanceOrg": 210000.0,
        "newbalanceOrig": 10000.0,
        "nameDest": "C9000000003",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 200000.0,
    },
]


# Verifies that /health returns 200 and the required fields are present
async def test_health(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "model_version" in body
    assert "operating_threshold" in body
    assert body["status"] == "ok"
    assert isinstance(body["operating_threshold"], float)


# Verifies that a drain-pattern CASH_OUT scores above 0.5 and lands in BLOCK or REVIEW
async def test_score_fraud_like(client: AsyncClient, model_available: bool):
    response = await client.post("/score", json=FRAUD_LIKE_TX)
    if not model_available:
        assert response.status_code == 503
        return
    assert response.status_code == 200
    body = response.json()
    assert body["score"] > 0.5
    assert body["decision"] in ("BLOCK", "REVIEW")
    assert "decision_id" in body
    assert len(body["reasons"]) > 0


# Verifies that a balanced small PAYMENT scores below 0.5 and is APPROVED
async def test_score_legit_like(client: AsyncClient, model_available: bool):
    response = await client.post("/score", json=LEGIT_LIKE_TX)
    if not model_available:
        assert response.status_code == 503
        return
    assert response.status_code == 200
    body = response.json()
    assert body["score"] < 0.5
    assert body["decision"] == "APPROVE"


# Verifies that a scored decision is retrievable from the audit log by decision_id
async def test_audit_trail(client: AsyncClient, model_available: bool):
    if not model_available:
        pytest.skip("Model not available — skipping audit trail test")
    score_response = await client.post("/score", json=FRAUD_LIKE_TX)
    assert score_response.status_code == 200
    decision_id = score_response.json()["decision_id"]

    audit_response = await client.get(f"/decisions/{decision_id}")
    assert audit_response.status_code == 200
    record = audit_response.json()
    assert record["decision_id"] == decision_id
    assert record["transaction_id"] == FRAUD_LIKE_TX["transaction_id"]
    assert "score" in record
    assert "decision" in record
    assert "top_reasons" in record
    assert "latency_ms" in record


# Verifies that /batch returns exactly 3 results and a correctly shaped summary
async def test_batch(client: AsyncClient, model_available: bool):
    response = await client.post("/batch", json=BATCH_TXS)
    if not model_available:
        assert response.status_code == 503
        return
    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert "summary" in body
    assert len(body["results"]) == 3
    summary = body["summary"]
    assert summary["total"] == 3
    assert summary["blocked"] + summary["review"] + summary["approved"] == 3


# Verifies that /monitor/spike returns the required spike_detected and ewma_rate fields
async def test_monitor_spike(client: AsyncClient):
    response = await client.get("/monitor/spike")
    assert response.status_code == 200
    body = response.json()
    assert "spike_detected" in body
    assert "ewma_rate" in body
    assert isinstance(body["spike_detected"], bool)
    assert isinstance(body["ewma_rate"], float)
