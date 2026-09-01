#!/usr/bin/env bash

BASE_URL="http://localhost:8000"

echo "============================================"
echo " RazorSentry — Demo curl commands"
echo " Make sure the service is running:"
echo "   make run"
echo "============================================"
echo ""

echo "--- 1. Health check ---"
curl -s -X GET "${BASE_URL}/health" | python3 -m json.tool
echo ""

echo "--- 2. High-risk CASH_OUT (expect BLOCK or REVIEW) ---"
SCORE_RESP=$(curl -s -X POST "${BASE_URL}/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "demo_fraud_001",
    "step": 1,
    "type": "CASH_OUT",
    "amount": 485000.0,
    "nameOrig": "C1234567890",
    "oldbalanceOrg": 490000.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C9876543210",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 485000.0
  }')
echo "$SCORE_RESP" | python3 -m json.tool
DECISION_ID=$(echo "$SCORE_RESP" | python3 -c "import sys, json; print(json.load(sys.stdin).get('decision_id', ''))")
DECISION_TYPE=$(echo "$SCORE_RESP" | python3 -c "import sys, json; print(json.load(sys.stdin).get('decision', ''))")
echo ""

echo "--- 3. Low-risk PAYMENT (expect APPROVE) ---"
curl -s -X POST "${BASE_URL}/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "demo_legit_001",
    "step": 2,
    "type": "PAYMENT",
    "amount": 1200.0,
    "nameOrig": "C1111111111",
    "oldbalanceOrg": 50000.0,
    "newbalanceOrig": 48800.0,
    "nameDest": "M2222222222",
    "oldbalanceDest": 10000.0,
    "newbalanceDest": 11200.0
  }' | python3 -m json.tool
echo ""

echo "--- 4. Retrieve decision from audit log (using decision_id: ${DECISION_ID}) ---"
curl -s -X GET "${BASE_URL}/decisions/${DECISION_ID}" | python3 -m json.tool
echo ""

echo "--- 5. Groq LLM Analyst Note (for REVIEW items) ---"
if [ "$DECISION_TYPE" = "REVIEW" ]; then
  curl -s -X GET "${BASE_URL}/analyst/note/${DECISION_ID}" | python3 -m json.tool
else
  echo "Decision was $DECISION_TYPE — analyst notes only generated for REVIEW decisions"
fi
echo ""

echo "--- 6. Fraud spike monitor ---"
curl -s -X GET "${BASE_URL}/monitor/spike" | python3 -m json.tool
echo ""

echo "--- 7. Recent decisions (last 10) ---"
curl -s -X GET "${BASE_URL}/decisions?limit=10" | python3 -m json.tool
echo ""

echo "=== Razorpay Webhook (payment.failed event) ==="
curl -s -X POST http://localhost:8000/webhook/razorpay \
  -H "Content-Type: application/json" \
  -d '{
    "entity": "event",
    "account_id": "acc_test123",
    "event": "payment.failed",
    "contains": ["payment"],
    "payload": {
      "payment": {
        "entity": {
          "id": "pay_test_fraud001",
          "amount": 18900000,
          "method": "wallet",
          "contact": "C999888777",
          "email": "M_MERCHANT_001",
          "status": "failed"
        }
      }
    }
  }' | python3 -m json.tool
echo ""

echo "=== Async Scoring ==="
JOB_RESPONSE=$(curl -s -X POST http://localhost:8000/score/async \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "ASYNC_DEMO_001",
    "step": 1,
    "type": "CASH_OUT",
    "amount": 175000.0,
    "nameOrig": "C555666777",
    "oldbalanceOrg": 175000.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C888999000",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 175000.0
  }')
echo $JOB_RESPONSE | python3 -m json.tool
JOB_ID=$(echo $JOB_RESPONSE | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo ""
echo "=== Polling for async result (waiting 4 seconds) ==="
sleep 4
curl -s http://localhost:8000/score/result/$JOB_ID | python3 -m json.tool

echo ""
echo "=== Dashboard Stats ==="
curl -s http://localhost:8000/dashboard/stats | python3 -m json.tool | head -20

echo ""
echo "=== Open dashboard in browser ==="
echo "http://localhost:8000/dashboard"

echo ""
echo "============================================"
echo " Demo complete."
echo "============================================"
