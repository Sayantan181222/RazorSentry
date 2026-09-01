# Locust load test — simulates concurrent users hitting RazorSentry scoring endpoints

import random
from locust import HttpUser, between, task


FRAUD_TXN = {
    "transaction_id": "LOAD_TEST_FRAUD",
    "step": 1,
    "type": "CASH_OUT",
    "amount": 185000.0,
    "nameOrig": "C123456789",
    "oldbalanceOrg": 185000.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C987654321",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 185000.0,
}

LEGIT_TXN = {
    "transaction_id": "LOAD_TEST_LEGIT",
    "step": 1,
    "type": "PAYMENT",
    "amount": 1200.0,
    "nameOrig": "C111111111",
    "oldbalanceOrg": 50000.0,
    "newbalanceOrig": 48800.0,
    "nameDest": "M999999999",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
}


# Simulates a fraud operations user hitting the sync score and health endpoints
class FraudOpsUser(HttpUser):
    wait_time = between(0.1, 0.5)

    # Scores a mixed transaction — 80% legit 20% fraud matching real-world distribution
    @task(8)
    def score_legit(self):
        txn = LEGIT_TXN.copy()
        txn["transaction_id"] = f"LOAD_LEGIT_{random.randint(1, 999999)}"
        txn["amount"] = round(random.uniform(200, 30000), 2)
        self.client.post(
            "/score",
            json=txn,
            name="/score [legit]",
        )

    # Scores a fraud-pattern transaction to test BLOCK path latency
    @task(2)
    def score_fraud(self):
        txn = FRAUD_TXN.copy()
        txn["transaction_id"] = f"LOAD_FRAUD_{random.randint(1, 999999)}"
        self.client.post(
            "/score",
            json=txn,
            name="/score [fraud]",
        )

    # Checks health endpoint to simulate monitoring traffic
    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
