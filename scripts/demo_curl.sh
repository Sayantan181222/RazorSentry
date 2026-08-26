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
curl -s -X POST "${BASE_URL}/score" \
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
  }' | python3 -m json.tool
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

echo "--- 4. Retrieve decision from audit log ---"
echo "Replace DECISION_ID below with a real UUID from step 2 or 3 above"
DECISION_ID="paste-decision-id-here"
curl -s -X GET "${BASE_URL}/decisions/${DECISION_ID}" | python3 -m json.tool
echo ""

echo "--- 5. Fraud spike monitor ---"
curl -s -X GET "${BASE_URL}/monitor/spike" | python3 -m json.tool
echo ""

echo "--- 6. Recent decisions (last 10) ---"
curl -s -X GET "${BASE_URL}/decisions?limit=10" | python3 -m json.tool
echo ""

echo "============================================"
echo " Demo complete."
echo "============================================"
