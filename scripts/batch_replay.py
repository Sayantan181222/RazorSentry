import csv
import os
import sys

import httpx
import pandas as pd

TEST_PATH = os.getenv("TEST_PATH", "data/test.parquet")
SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:8000")
REPLAY_ROWS = 1000
RESULTS_PATH = os.path.join("reports", "batch_replay_results.csv")

SCORE_ENDPOINT = f"{SERVICE_URL}/score"

DECISION_COLORS = {
    "BLOCK": "\033[91m",
    "REVIEW": "\033[93m",
    "APPROVE": "\033[92m",
}
RESET = "\033[0m"


# Loads the test parquet and returns the first N rows as a DataFrame
def load_sample(path: str, n: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.head(n).reset_index(drop=True)


# Converts a single DataFrame row to the JSON payload expected by POST /score
def row_to_payload(row: pd.Series) -> dict:
    return {
        "transaction_id": str(row.get("transaction_id", f"replay_{row.name}")),
        "step": int(row["step"]),
        "type": str(row["type"]),
        "amount": float(row["amount"]),
        "nameOrig": str(row["nameOrig"]),
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),
        "nameDest": str(row["nameDest"]),
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"]),
    }


# Posts a single transaction payload to /score and returns the parsed JSON response
def post_score(client: httpx.Client, payload: dict) -> dict:
    response = client.post(SCORE_ENDPOINT, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()


# Formats a single result line for terminal output with ANSI colour coding
def format_result_line(result: dict, idx: int) -> str:
    decision = result.get("decision", "UNKNOWN")
    score = result.get("score", 0.0)
    reasons = result.get("reasons", [])
    first_reason = reasons[0] if reasons else "n/a"
    latency = result.get("latency_ms", 0.0)
    txn_id = result.get("transaction_id", f"row_{idx}")
    color = DECISION_COLORS.get(decision, "")
    return (
        f"[{txn_id}] {color}{decision}{RESET} | "
        f"score={score:.4f} | "
        f"reason={first_reason} | "
        f"latency={latency:.1f}ms"
    )


# Prints the final summary banner after all transactions are processed
def print_summary(results: list[dict], total: int) -> None:
    blocked = sum(1 for r in results if r.get("decision") == "BLOCK")
    review = sum(1 for r in results if r.get("decision") == "REVIEW")
    approved = sum(1 for r in results if r.get("decision") == "APPROVE")
    latencies = [r.get("latency_ms", 0.0) for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    print("\n" + "=" * 60)
    print("RazorSentry — Batch Replay Summary")
    print("=" * 60)
    print(f"Total      : {total:,}")
    print(f"Blocked    : {blocked:,}  ({blocked/total*100:.1f}%)")
    print(f"Review     : {review:,}  ({review/total*100:.1f}%)")
    print(f"Approved   : {approved:,}  ({approved/total*100:.1f}%)")
    print(f"Avg latency: {avg_latency:.1f}ms")
    print("=" * 60)


# Saves all replay results to a CSV file in the reports directory
def save_results_csv(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not results:
        return
    fieldnames = [
        "transaction_id",
        "decision_id",
        "score",
        "decision",
        "reasons",
        "latency_ms",
        "model_version",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["reasons"] = "; ".join(r.get("reasons", []))
            writer.writerow(row)


# Orchestrates loading, streaming replay, terminal output, summary, and CSV save
def main() -> None:
    print(f"Loading {REPLAY_ROWS} rows from {TEST_PATH} ...")
    df = load_sample(TEST_PATH, REPLAY_ROWS)
    print(f"Loaded {len(df):,} rows. Streaming to {SCORE_ENDPOINT} ...\n")

    results = []
    errors = 0

    with httpx.Client() as client:
        for idx, row in df.iterrows():
            payload = row_to_payload(row)
            try:
                result = post_score(client, payload)
                results.append(result)
                print(format_result_line(result, idx))
            except httpx.HTTPStatusError as e:
                errors += 1
                print(f"[row_{idx}] ERROR {e.response.status_code} — {e.response.text[:80]}")
            except httpx.RequestError as e:
                errors += 1
                print(f"[row_{idx}] CONNECTION ERROR — {str(e)[:80]}")
                print("Is the service running? Start it with: make run")
                sys.exit(1)

    if errors:
        print(f"\n{errors} request(s) failed — see above for details.")

    print_summary(results, len(df))
    save_results_csv(results, RESULTS_PATH)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
