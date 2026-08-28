import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.features import build_features, get_feature_columns

TEST_PATH = os.getenv("TEST_PATH", "data/test.parquet")
MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm_model.pkl")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "models/threshold.txt")
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")

BLOCK_THRESHOLD = 0.5
FP_COST_DEFAULT = 150.0
FP_COST_SENSITIVITY = [50.0, 150.0, 300.0]

TOP_FP_COLUMNS = [
    "transaction_id",
    "score",
    "amount",
    "type",
    "balance_error_orig",
    "dest_in_degree_1h",
    "drain_flag",
]


# Loads the test parquet file and returns a DataFrame
def load_test(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


# Loads and returns the serialised LightGBM model from disk
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# Reads the operating threshold from a plain-text file
def load_threshold(path: str) -> float:
    with open(path, "r") as f:
        return float(f.read().strip())


# Applies the three-tier decision policy to a probability array
def apply_policy(y_prob: np.ndarray, operating_threshold: float) -> np.ndarray:
    decisions = np.where(
        y_prob >= BLOCK_THRESHOLD,
        "BLOCK",
        np.where(y_prob >= operating_threshold, "REVIEW", "APPROVE"),
    )
    return decisions


# Converts BLOCK/REVIEW decisions to 1 and APPROVE to 0 for sklearn metrics
def decisions_to_binary(decisions: np.ndarray) -> np.ndarray:
    return (decisions != "APPROVE").astype(int)


# Computes net rupee savings given prediction arrays and cost parameters
def compute_net_savings(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    avg_fraud_loss: float,
    fp_cost: float,
) -> dict:
    fp = int(((y_pred_binary == 1) & (y_true == 0)).sum())
    fn = int(((y_pred_binary == 0) & (y_true == 1)).sum())
    total_frauds = int(y_true.sum())
    total_cost = fp * fp_cost + fn * avg_fraud_loss
    net_savings = total_frauds * avg_fraud_loss - total_cost
    return {
        "false_positives": fp,
        "false_negatives": fn,
        "total_frauds": total_frauds,
        "total_cost_inr": float(total_cost),
        "net_savings_inr": float(net_savings),
    }


# Sweeps thresholds from 0.01 to 0.99 and returns net savings at each point
def sweep_cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    avg_fraud_loss: float,
    fp_cost: float,
) -> list[tuple[float, float]]:
    thresholds = np.arange(0.01, 1.00, 0.01)
    results = []
    total_frauds = int(y_true.sum())
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        net_savings = total_frauds * avg_fraud_loss - (fp * fp_cost + fn * avg_fraud_loss)
        results.append((float(t), float(net_savings)))
    return results


# Plots and saves the precision-recall curve to the reports directory
def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pr_auc: float,
    out_dir: str,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("RazorSentry — Precision-Recall Curve (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curve.png"))
    plt.close()


# Plots and saves net savings vs threshold as the cost curve
def plot_cost_curve(curve_data: list[tuple[float, float]], out_dir: str) -> None:
    thresholds = [p[0] for p in curve_data]
    savings = [p[1] for p in curve_data]
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, savings, color="darkorange")
    plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    plt.xlabel("Threshold")
    plt.ylabel("Net Savings (INR)")
    plt.title("RazorSentry — Net Savings vs Threshold (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cost_curve.png"))
    plt.close()


# Plots and saves the confusion matrix at the operating threshold
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    out_dir: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("RazorSentry — Confusion Matrix at Operating Threshold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()


# Saves the top 10 highest-scoring false positive rows to a CSV
def save_top_fp_cases(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: str,
) -> None:
    df = test_df.copy().reset_index(drop=True)
    df["score"] = y_prob
    fp_mask = (y_prob >= 0.5) & (y_true == 0)
    fp_df = df[fp_mask].nlargest(10, "score")
    cols_present = [c for c in TOP_FP_COLUMNS if c in fp_df.columns]
    fp_df[cols_present].to_csv(
        os.path.join(out_dir, "top_fp_cases.csv"), index=False
    )


# Prints a sensitivity table showing net savings under different fp_cost assumptions
def print_sensitivity_table(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    avg_fraud_loss: float,
    fp_costs: list[float],
) -> None:
    print("\n=== Sensitivity Table: Net Savings vs FP Cost ===")
    print(f"{'FP Cost (INR)':>16} | {'False Positives':>16} | {'False Negatives':>16} | {'Net Savings (INR)':>18}")
    print("-" * 74)
    total_frauds = int(y_true.sum())
    fp_count = int(((y_pred_binary == 1) & (y_true == 0)).sum())
    fn_count = int(((y_pred_binary == 0) & (y_true == 1)).sum())
    for fp_cost in fp_costs:
        total_cost = fp_count * fp_cost + fn_count * avg_fraud_loss
        net_savings = total_frauds * avg_fraud_loss - total_cost
        print(
            f"{fp_cost:>16.0f} | {fp_count:>16,} | {fn_count:>16,} | {net_savings:>18,.2f}"
        )
    print()


# Saves the full evaluation metrics dictionary to reports/metrics.json
def save_metrics(metrics: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


# Orchestrates the full evaluation pipeline and prints all results
def main() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading test data ...")
    test_raw = load_test(TEST_PATH)

    print("Building features ...")
    test_df = build_features(test_raw)

    feat_cols = get_feature_columns()
    X_test = test_df[feat_cols]
    y_true = test_df["isFraud"].astype(int).values

    print("Loading model and threshold ...")
    model = load_model(MODEL_PATH)
    operating_threshold = load_threshold(THRESHOLD_PATH)
    print(f"Operating threshold: {operating_threshold:.4f}")

    print("Scoring test set ...")
    y_prob = model.predict_proba(X_test)[:, 1]

    decisions = apply_policy(y_prob, operating_threshold)
    y_pred_binary = decisions_to_binary(decisions)

    pr_auc = float(average_precision_score(y_true, y_prob))
    precision = float(precision_score(y_true, y_pred_binary, zero_division=0))
    recall = float(recall_score(y_true, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))

    total_txns = len(y_true)
    total_frauds = int(y_true.sum())
    fp_count = int(((y_pred_binary == 1) & (y_true == 0)).sum())
    fn_count = int(((y_pred_binary == 0) & (y_true == 1)).sum())
    fp_rate = fp_count / max(int((y_true == 0).sum()), 1)

    avg_fraud_loss = float(np.median(test_df.loc[y_true == 1, "amount"].values))
    cost_info = compute_net_savings(y_true, y_pred_binary, avg_fraud_loss, FP_COST_DEFAULT)

    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp_cm, fn_cm, tp = cm.ravel() if cm.shape == (2, 2) else (0, fp_count, fn_count, 0)

    metrics = {
        "operating_threshold": operating_threshold,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": fp_count,
        "false_positive_rate": round(fp_rate, 6),
        "false_negatives": fn_count,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "net_savings_inr": cost_info["net_savings_inr"],
        "avg_fraud_loss_inr": avg_fraud_loss,
        "fp_cost_per_txn_inr": FP_COST_DEFAULT,
        "total_test_txns": total_txns,
        "total_test_frauds": total_frauds,
        "decision_counts": {
            "BLOCK": int((decisions == "BLOCK").sum()),
            "REVIEW": int((decisions == "REVIEW").sum()),
            "APPROVE": int((decisions == "APPROVE").sum()),
        },
    }
    save_metrics(metrics, REPORTS_DIR)

    print("\n=== Evaluation Results ===")
    print(f"PR-AUC              : {pr_auc:.4f}")
    print(f"Precision           : {precision:.4f}")
    print(f"Recall              : {recall:.4f}")
    print(f"F1                  : {f1:.4f}")
    print(f"False Positives     : {fp_count:,}  (rate: {fp_rate:.4%})")
    print(f"False Negatives     : {fn_count:,}")
    print(f"Avg Fraud Loss      : ₹{avg_fraud_loss:,.2f}")
    print(f"Net Savings (INR)   : ₹{cost_info['net_savings_inr']:,.2f}")
    print(f"Decision breakdown  : BLOCK={metrics['decision_counts']['BLOCK']:,}  REVIEW={metrics['decision_counts']['REVIEW']:,}  APPROVE={metrics['decision_counts']['APPROVE']:,}")

    print_sensitivity_table(y_true, y_pred_binary, avg_fraud_loss, FP_COST_SENSITIVITY)

    print("Generating plots ...")
    curve_data = sweep_cost_curve(y_true, y_prob, avg_fraud_loss, FP_COST_DEFAULT)
    plot_pr_curve(y_true, y_prob, pr_auc, REPORTS_DIR)
    plot_cost_curve(curve_data, REPORTS_DIR)
    plot_confusion_matrix(y_true, y_pred_binary, REPORTS_DIR)

    print("Saving top FP cases ...")
    save_top_fp_cases(test_df, y_true, y_prob, REPORTS_DIR)

    print(f"\nAll reports written to {REPORTS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
