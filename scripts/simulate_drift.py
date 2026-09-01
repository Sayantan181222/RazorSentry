# Simulates a festival-season fraud ring to trigger PSI drift detection visibly

import httpx
import time
import random
import json

BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}


# Sends a batch of transactions to /score and prints each decision
def send_batch(transactions: list[dict], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    decisions = {"BLOCK": 0, "REVIEW": 0, "APPROVE": 0}
    total_amount = 0
    for i, tx in enumerate(transactions):
        response = httpx.post(f"{BASE_URL}/score", json=tx, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            result = response.json()
            dec = result["decision"]
            decisions[dec] += 1
            total_amount += tx["amount"]
            print(f"  [{i+1:02d}] {dec:7s} | score={result['score']:.4f} | ₹{tx['amount']:>12,.0f} | {tx['type']}")
        else:
            print(f"  [{i+1:02d}] ERROR {response.status_code}")
        time.sleep(0.05)
    print(f"\n  Summary: BLOCK={decisions['BLOCK']} REVIEW={decisions['REVIEW']} APPROVE={decisions['APPROVE']}")
    print(f"  Total amount: ₹{total_amount:,.0f}")


# Generates normal PaySim-like transactions matching training distribution
def make_normal_transactions(n: int) -> list[dict]:
    types = ["PAYMENT", "PAYMENT", "PAYMENT", "CASH_IN", "DEBIT"]
    txns = []
    for i in range(n):
        amount = random.uniform(500, 25000)
        balance = random.uniform(amount * 1.5, amount * 10)
        txns.append({
            "transaction_id": f"NORMAL_{i:04d}_{int(time.time())}",
            "step": 1,
            "type": random.choice(types),
            "amount": round(amount, 2),
            "nameOrig": f"C{random.randint(100000000, 999999999)}",
            "oldbalanceOrg": round(balance, 2),
            "newbalanceOrig": round(balance - amount, 2),
            "nameDest": f"M{random.randint(100000000, 999999999)}",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
        })
    return txns


# Generates drifted transactions simulating a festival-season coordinated fraud ring
def make_drifted_transactions(n: int) -> list[dict]:
    # Drift pattern: all CASH_OUT, amounts 10x-50x larger than normal,
    # accounts draining completely — simulates coordinated mule network
    # This is the Diwali/IPL season fraud pattern seen in Indian payments
    MULE_DESTINATIONS = [
        f"C{random.randint(100000000, 999999999)}" for _ in range(5)
    ]
    txns = []
    for i in range(n):
        amount = random.uniform(450000, 950000)
        txns.append({
            "transaction_id": f"DRIFT_{i:04d}_{int(time.time())}",
            "step": 1,
            "type": "CASH_OUT",
            "amount": round(amount, 2),
            "nameOrig": f"C{random.randint(100000000, 999999999)}",
            "oldbalanceOrg": round(amount, 2),
            "newbalanceOrig": 0.0,
            "nameDest": random.choice(MULE_DESTINATIONS),
            "oldbalanceDest": 0.0,
            "newbalanceDest": round(amount, 2),
        })
    return txns


# Fetches and prints the current drift status from the monitor endpoint
def check_drift_status(label: str) -> None:
    print(f"\n--- Drift Status: {label} ---")
    response = httpx.get(f"{BASE_URL}/monitor/drift", timeout=10)
    data = response.json()
    if not data.get("drift_checked"):
        print(f"  Not checked: {data.get('reason', 'unknown')}")
        return
    print(f"  Max PSI: {data.get('max_psi', 0):.4f}")
    print(f"  Alert:   {data.get('alert', False)}")
    print(f"  Warn:    {data.get('warn', False)}")
    features = data.get("feature_psi", {})
    for feat, psi in sorted(features.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(psi * 20)
        flag = " ← DRIFT ALERT" if psi > 0.2 else (" ← WARN" if psi > 0.1 else "")
        print(f"  {feat:25s} PSI={psi:.4f}  {bar}{flag}")
    if data.get("alert_features"):
        print(f"\n  ⚠️  Alert features: {', '.join(data['alert_features'])}")


# Checks spike status from the monitor endpoint
def check_spike_status() -> None:
    print("\n--- Spike Status ---")
    response = httpx.get(f"{BASE_URL}/monitor/spike", timeout=10)
    data = response.json()
    print(f"  Spike detected: {data.get('spike_detected', False)}")
    print(f"  EWMA rate:      {data.get('ewma_rate', 0)*100:.2f}%")


if __name__ == "__main__":
    print("\n🛡️  RazorSentry — Data Drift Simulation")
    print("Scenario: Festival-season coordinated fraud ring (Diwali pattern)")
    print("Normal training distribution: mixed types, ₹500-25,000 amounts")
    print("Drifted distribution: all CASH_OUT, ₹4,50,000-9,50,000 amounts\n")

    print("Step 1: Sending 50 NORMAL transactions to establish baseline...")
    normal_txns = make_normal_transactions(50)
    send_batch(normal_txns, "NORMAL TRANSACTIONS — Training Distribution")

    check_drift_status("After normal transactions (should be stable)")
    check_spike_status()

    print("\n\nStep 2: Sending 50 DRIFTED transactions (fraud ring pattern)...")
    print("Watch the dashboard at http://localhost:8000/dashboard")
    drifted_txns = make_drifted_transactions(50)
    send_batch(drifted_txns, "DRIFTED TRANSACTIONS — Festival Fraud Ring Pattern")

    print("\n\nStep 3: Checking drift detection...")
    check_drift_status("After drifted transactions (should show ALERT)")
    check_spike_status()

    print("\n\n✅ Simulation complete.")
    print("Open http://localhost:8000/dashboard to see the drift alert on the dashboard.")
    print("The Feature Drift Monitor should now show 🔴 DRIFT ALERT (PSI > 0.2)")
