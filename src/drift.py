# PSI drift detection is LLM-free by design — pure statistics, interpretable, fast

import json
import os
import numpy as np
import pandas as pd

PSI_WARN_THRESHOLD = 0.1
PSI_ALERT_THRESHOLD = 0.2
REFERENCE_STATS_PATH = os.getenv("REFERENCE_STATS_PATH", "models/reference_stats.json")


# Computes Population Stability Index between reference and current distributions
def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref_counts, bin_edges = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    ref_pct = (ref_counts + 1e-6) / len(reference)
    cur_pct = (cur_counts + 1e-6) / len(current)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 6)


# Saves mean and std of each feature from the training set as reference statistics
def save_reference_stats(df_train: pd.DataFrame, feature_cols: list) -> None:
    stats = {}
    for col in feature_cols:
        values = df_train[col].dropna().values
        stats[col] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values_sample": values[:500].tolist(),
        }
    os.makedirs(os.path.dirname(REFERENCE_STATS_PATH) or ".", exist_ok=True)
    with open(REFERENCE_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)


# Loads the saved reference statistics from training time
def load_reference_stats() -> dict:
    if not os.path.exists(REFERENCE_STATS_PATH):
        return {}
    with open(REFERENCE_STATS_PATH) as f:
        return json.load(f)


# Computes PSI for each feature in the incoming batch against training reference
def check_drift(incoming_df: pd.DataFrame, feature_cols: list) -> dict:
    ref = load_reference_stats()
    if not ref:
        return {"drift_checked": False, "reason": "No reference stats found — run train.py first"}

    results = {}
    max_psi = 0.0
    alert_features = []

    for col in feature_cols:
        if col not in ref or col not in incoming_df.columns:
            continue
        reference_sample = np.array(ref[col]["values_sample"])
        current_sample = incoming_df[col].dropna().values
        if len(current_sample) < 10:
            continue
        psi = compute_psi(reference_sample, current_sample)
        results[col] = psi
        if psi > max_psi:
            max_psi = psi
        if psi > PSI_ALERT_THRESHOLD:
            alert_features.append(col)

    return {
        "drift_checked": True,
        "max_psi": round(max_psi, 6),
        "alert": max_psi > PSI_ALERT_THRESHOLD,
        "warn": max_psi > PSI_WARN_THRESHOLD,
        "alert_features": alert_features,
        "feature_psi": results,
        "interpretation": {
            "0.0-0.1": "No significant change",
            "0.1-0.2": "Moderate change — monitor",
            ">0.2": "Significant shift — consider retraining",
        },
    }
