import os
# pyrefly: ignore [missing-import]
import pytest
from httpx import AsyncClient

FRAUD_TXN = {
    "transaction_id": "txn_fraud_001",
    "step": 1,
    "type": "CASH_OUT",
    "amount": 180000.0,
    "nameOrig": "C123456789",
    "oldbalanceOrg": 180000.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C987654321",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 180000.0,
}

LEGIT_TXN = {
    "transaction_id": "txn_legit_001",
    "step": 1,
    "type": "PAYMENT",
    "amount": 500.0,
    "nameOrig": "C111111111",
    "oldbalanceOrg": 10000.0,
    "newbalanceOrig": 9500.0,
    "nameDest": "M999999999",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
}


# Tests that the health endpoint returns 200 with required fields
@pytest.mark.anyio
async def test_health(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_version" in data


# Tests that the score endpoint returns a valid response structure
@pytest.mark.anyio
async def test_score_returns_structure(client: AsyncClient, model_available: bool):
    response = await client.post("/score", json=FRAUD_TXN)
    if not model_available:
        assert response.status_code == 503
        return
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert "score" in data
    assert "decision_id" in data
    assert data["decision"] in ("BLOCK", "REVIEW", "APPROVE")


# Tests that a batch of transactions returns correct count and summary
@pytest.mark.anyio
async def test_batch_returns_summary(client: AsyncClient, model_available: bool):
    response = await client.post("/batch", json=[FRAUD_TXN, LEGIT_TXN])
    if not model_available:
        assert response.status_code == 503
        return
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "summary" in data
    assert len(data["results"]) == 2


# Tests that audit trail returns a record after scoring
@pytest.mark.anyio
async def test_audit_trail(client: AsyncClient, model_available: bool):
    if not model_available:
        pytest.skip("Model not available in CI environment")
    score_resp = await client.post("/score", json=LEGIT_TXN)
    assert score_resp.status_code == 200
    decision_id = score_resp.json()["decision_id"]
    audit_resp = await client.get(f"/decisions/{decision_id}")
    assert audit_resp.status_code == 200
    assert audit_resp.json()["decision_id"] == decision_id


# Tests that the spike monitor endpoint returns required fields
@pytest.mark.anyio
async def test_monitor_spike(client: AsyncClient):
    response = await client.get("/monitor/spike")
    assert response.status_code == 200
    data = response.json()
    assert "spike_detected" in data
    assert "ewma_rate" in data


# Tests that the webhook endpoint accepts a Razorpay-shaped payload
@pytest.mark.anyio
async def test_razorpay_webhook(client: AsyncClient, model_available: bool):
    payload = {
        "entity": "event",
        "account_id": "acc_test123",
        "event": "payment.failed",
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_test001",
                    "amount": 18900000,
                    "method": "wallet",
                    "contact": "C999888777",
                    "email": "M_MERCHANT_001",
                    "status": "failed",
                }
            }
        },
    }
    response = await client.post("/webhook/razorpay", json=payload)
    if not model_available:
        assert response.status_code == 503
        return
    assert response.status_code == 200
    data = response.json()
    assert "razorsentry_decision" in data
    assert "webhook_event" in data


# Tests that the audit log stores a blinded transaction_id not the raw value
@pytest.mark.anyio
async def test_pii_blinding(client: AsyncClient, model_available: bool):
    if not model_available:
        pytest.skip("Model not available in CI environment")
    txn = {
        "transaction_id": "SENSITIVE_ACCOUNT_12345",
        "step": 1,
        "type": "PAYMENT",
        "amount": 500.0,
        "nameOrig": "C111111111",
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 9500.0,
        "nameDest": "M999999999",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    score_resp = await client.post("/score", json=txn)
    assert score_resp.status_code == 200
    decision_id = score_resp.json()["decision_id"]
    audit_resp = await client.get(f"/decisions/{decision_id}")
    assert audit_resp.status_code == 200
    stored_txn_id = audit_resp.json()["transaction_id"]
    assert stored_txn_id != "SENSITIVE_ACCOUNT_12345", "Raw PII must not be stored in audit log"
    assert len(stored_txn_id) == 16, "Blinded ID must be 16 hex chars"


# Tests that the drift monitor endpoint returns required fields
@pytest.mark.anyio
async def test_monitor_drift(client: AsyncClient):
    response = await client.get("/monitor/drift")
    assert response.status_code == 200
    data = response.json()
    assert "drift_checked" in data


# Tests that invalid transaction type is rejected with 422
@pytest.mark.anyio
async def test_invalid_type_rejected(client: AsyncClient):
    bad_txn = {
        "transaction_id": "txn_bad_001",
        "step": 1,
        "type": "INVALID_TYPE",
        "amount": 500.0,
        "nameOrig": "C111111111",
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 9500.0,
        "nameDest": "M999999999",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    response = await client.post("/score", json=bad_txn)
    assert response.status_code == 422



