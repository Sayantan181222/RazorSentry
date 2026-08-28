import json
import os
import pickle

import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from src.drift import save_reference_stats
from src.features import build_features, get_feature_columns

TRAIN_PATH = os.getenv("TRAIN_PATH", "data/train.parquet")
TEST_PATH = os.getenv("TEST_PATH", "data/test.parquet")
MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm_model.pkl")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "models/threshold.txt")
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")

CASH_OUT_TYPE_STR = "CASH_OUT"
FP_COST_PER_TXN = 150.0
VAL_SPLIT = 0.10
RANDOM_STATE = 42


# Loads a parquet file and returns a DataFrame
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


# Applies rules to flag fraud: drain+CASH_OUT, balance error, or dest burst
def rules_baseline(df: pd.DataFrame) -> pd.Series:
    cash_out_mask = df["type"] == CASH_OUT_TYPE_STR
    rule1 = (df["drain_flag"] == 1) & cash_out_mask
    rule2 = df["balance_error_orig"] > 10_000
    rule3 = df["dest_in_degree_1h"] >= 5
    return (rule1 | rule2 | rule3).astype(int).reset_index(drop=True)


# Computes precision, recall, f1, and false-positive count from predictions
def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positives": fp,
    }


# Saves a dictionary as a formatted JSON file to the given path
def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# Computes scale_pos_weight as ratio of negatives to positives for class imbalance
def compute_scale_pos_weight(y: pd.Series) -> float:
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    return float(n_neg) / float(n_pos)


# Trains LightGBM with early stopping on a held-out validation slice of train data
def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
) -> LGBMClassifier:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[
            __import__("lightgbm").early_stopping(stopping_rounds=30, verbose=False),
            __import__("lightgbm").log_evaluation(period=-1),
        ],
    )
    return model, X_val, y_val


# Wraps the trained LightGBM model with isotonic calibration for interpretable probabilities
def calibrate_model(model, X_val: pd.DataFrame, y_val: pd.Series):
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated


# Plots reliability diagram comparing predicted probabilities to actual fraction positive
def plot_calibration_curve(model, X_test: pd.DataFrame, y_test: pd.Series, save_path: str) -> None:
    prob_true, prob_pred = calibration_curve(y_test, model.predict_proba(X_test)[:, 1], n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="RazorSentry")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Saves the trained model to disk using pickle
def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


# Saves the operating threshold as a plain text file
def save_threshold(threshold: float, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(str(threshold))


# Sweeps thresholds and returns the one that maximises net rupee savings
def find_operating_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    avg_fraud_loss: float,
) -> tuple[float, float]:
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_threshold = 0.5
    best_savings = -np.inf
    savings_list = []
    total_frauds = int(y_true.sum())

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        total_cost = fp * FP_COST_PER_TXN + fn * avg_fraud_loss
        net_savings = total_frauds * avg_fraud_loss - total_cost
        savings_list.append((t, net_savings))
        if net_savings > best_savings:
            best_savings = net_savings
            best_threshold = float(t)

    return best_threshold, savings_list


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
    plt.title("RazorSentry — Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curve.png"))
    plt.close()


# Plots and saves net savings vs threshold as the cost curve
def plot_cost_curve(savings_list: list[tuple[float, float]], out_dir: str) -> None:
    thresholds = [s[0] for s in savings_list]
    savings = [s[1] for s in savings_list]
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, savings, color="darkorange")
    plt.xlabel("Threshold")
    plt.ylabel("Net Savings (INR)")
    plt.title("RazorSentry — Net Savings vs Threshold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cost_curve.png"))
    plt.close()


# Plots and saves the confusion matrix at the operating threshold
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("RazorSentry — Confusion Matrix at Operating Threshold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()


# Saves the top 10 false-positive cases by fraud score to a CSV for transparency
def save_top_fp_cases(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: str,
) -> None:
    fp_mask = (y_prob >= 0.5) & (y_true == 0)
    fp_df = test_df.copy().reset_index(drop=True)
    fp_df["fraud_score"] = y_prob
    fp_df = fp_df[fp_mask].nlargest(10, "fraud_score")
    fp_df.to_csv(os.path.join(out_dir, "top_fp_cases.csv"), index=False)


# Orchestrates the full training pipeline: baseline, LightGBM, threshold sweep, reports
def main() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading data ...")
    train_raw = load_parquet(TRAIN_PATH)
    test_raw = load_parquet(TEST_PATH)

    print("Building features ...")
    train_df = build_features(train_raw)
    test_df = build_features(test_raw)

    feat_cols = get_feature_columns()
    save_reference_stats(train_df, feat_cols)
    print("Reference stats saved to models/reference_stats.json")
    X_train = train_df[feat_cols]
    y_train = train_df["isFraud"].astype(int).reset_index(drop=True)
    X_test = test_df[feat_cols]
    y_test = test_df["isFraud"].astype(int).reset_index(drop=True)

    print("Evaluating rules baseline ...")
    baseline_preds = rules_baseline(test_df)
    baseline_metrics = evaluate_predictions(y_test, baseline_preds)
    save_json(baseline_metrics, os.path.join(REPORTS_DIR, "baseline_metrics.json"))
    print(f"Baseline — precision={baseline_metrics['precision']:.4f}  recall={baseline_metrics['recall']:.4f}  f1={baseline_metrics['f1']:.4f}  FP={baseline_metrics['false_positives']}")

    spw = compute_scale_pos_weight(y_train)
    print(f"scale_pos_weight = {spw:.2f}")

    mlflow.set_experiment("razorsentry-fraud-detection")
    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "scale_pos_weight": spw,
            "val_split": VAL_SPLIT,
            "fp_cost_per_txn_inr": FP_COST_PER_TXN,
        })

        print("Training LightGBM ...")
        model, X_val, y_val = train_lgbm(X_train, y_train, spw)
        model = calibrate_model(model, X_val, y_val)

        val_prob = model.predict_proba(X_val)[:, 1]
        val_pr_auc = float(average_precision_score(y_val, val_prob))
        mlflow.log_metric("val_pr_auc", val_pr_auc)
        print(f"Val PR-AUC = {val_pr_auc:.4f}")

        y_prob = model.predict_proba(X_test)[:, 1]
        train_prob = model.predict_proba(X_train)[:, 1]
        train_pr_auc = float(average_precision_score(y_train, train_prob))
        test_pr_auc = float(average_precision_score(y_test, y_prob))
        mlflow.log_metric("train_pr_auc", train_pr_auc)
        mlflow.log_metric("test_pr_auc", test_pr_auc)
        print(f"Test PR-AUC = {test_pr_auc:.4f}")

        save_model(model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)

        avg_fraud_loss = float(test_df.loc[y_test == 1, "amount"].median())
        print(f"Avg fraud loss (median) = ₹{avg_fraud_loss:,.2f}")

        print("Sweeping thresholds ...")
        operating_threshold, savings_list = find_operating_threshold(
            y_test.values, y_prob, avg_fraud_loss
        )
        print(f"Operating threshold = {operating_threshold:.2f}")
        save_threshold(operating_threshold, THRESHOLD_PATH)
        mlflow.log_metric("operating_threshold", operating_threshold)

        y_pred_final = (y_prob >= operating_threshold).astype(int)
        fp = int(((y_pred_final == 1) & (y_test.values == 0)).sum())
        fn = int(((y_pred_final == 0) & (y_test.values == 1)).sum())
        total_frauds = int(y_test.sum())
        total_cost = fp * FP_COST_PER_TXN + fn * avg_fraud_loss
        net_savings = total_frauds * avg_fraud_loss - total_cost

        final_metrics = {
            "operating_threshold": operating_threshold,
            "precision": float(precision_score(y_test, y_pred_final, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_final, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_final, zero_division=0)),
            "pr_auc": test_pr_auc,
            "net_savings_inr": float(net_savings),
            "avg_fraud_loss_inr": avg_fraud_loss,
            "fp_cost_per_txn_inr": FP_COST_PER_TXN,
            "false_positives": fp,
            "false_negatives": fn,
            "baseline_precision": baseline_metrics["precision"],
            "baseline_recall": baseline_metrics["recall"],
            "total_test_txns": len(y_test),
            "total_test_frauds": total_frauds,
        }
        save_json(final_metrics, os.path.join(REPORTS_DIR, "metrics.json"))
        mlflow.log_metrics({
            "operating_precision": final_metrics["precision"],
            "operating_recall": final_metrics["recall"],
            "operating_f1": final_metrics["f1"],
            "net_savings_inr": net_savings,
        })

        print("Generating plots ...")
        plot_pr_curve(y_test.values, y_prob, test_pr_auc, REPORTS_DIR)
        plot_cost_curve(savings_list, REPORTS_DIR)
        plot_confusion_matrix(y_test.values, y_pred_final, REPORTS_DIR)
        plot_calibration_curve(model, X_test, y_test, os.path.join(REPORTS_DIR, "calibration_curve.png"))
        mlflow.log_artifacts(REPORTS_DIR)

        print("Saving top FP cases ...")
        save_top_fp_cases(test_df, y_test.values, y_prob, REPORTS_DIR)

        print("\n=== Final Metrics ===")
        print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
