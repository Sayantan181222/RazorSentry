# This module defines the scorable unit of work that RQ executes in a background worker process

import os
import pickle
import sys

import numpy as np
import shap
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audit import log_decision
from src.features import build_features, get_feature_columns
from src.privacy import blind_identifier

MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm_model.pkl")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "models/threshold.txt")
MODEL_VERSION = "lgbm_v1"
BLOCK_THRESHOLD = 0.5

REASON_MAP = {
    "balance_error_orig": "Accounting inconsistency in sender balance",
    "balance_error_dest": "Accounting inconsistency in receiver balance",
    "drain_flag": "Account being drained — amount near full balance",
    "zero_orig_after": "Sender account emptied to zero after transaction",
    "type_encoded": "Transaction type is high-risk",
    "amount_log": "Unusually large transaction amount",
    "orig_txn_count_1h": "Sender made multiple transactions in the last step",
    "orig_txn_sum_1h": "Sender moved large total volume in the last step",
    "dest_in_degree_1h": "Multiple senders hit this account in the last step",
    "high_amount_flag": "Transaction exceeds high-value amount threshold",
}

_model = None
_explainer = None
_operating_threshold = 0.35
_feature_columns = []


# Loads model, explainer, and threshold once per worker process and caches them globally
def _load_model_once():
    global _model, _explainer, _operating_threshold, _feature_columns
    if _model is not None:
        return
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    tree_model = (
        _model.estimator
        if hasattr(_model, "estimator")
        else (
            _model.calibrated_classifiers_[0].estimator
            if hasattr(_model, "calibrated_classifiers_")
            else _model
        )
    )
    _explainer = shap.TreeExplainer(tree_model)
    with open(THRESHOLD_PATH) as f:
        _operating_threshold = float(f.read().strip())
    _feature_columns = get_feature_columns()


# Applies the three-tier decision policy and returns BLOCK, REVIEW, or APPROVE
def _apply_policy(score: float) -> str:
    if score >= BLOCK_THRESHOLD:
        return "BLOCK"
    if score >= _operating_threshold:
        return "REVIEW"
    return "APPROVE"


# Sorts SHAP values by magnitude and returns the top N human-readable reason strings
def _top_reasons(shap_values: np.ndarray, feature_cols: list, top_n: int = 3) -> list:
    pairs = sorted(
        zip(feature_cols, shap_values.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return [REASON_MAP.get(feat, feat) for feat, _ in pairs[:top_n]]


# Scores one transaction dict and writes the decision to PostgreSQL — called by RQ worker
def score_transaction_job(tx_dict: dict) -> dict:
    import time
    import pandas as pd

    _load_model_once()

    t0 = time.perf_counter()
    df = pd.DataFrame([tx_dict])
    df_feat = build_features(df)
    X = df_feat[_feature_columns]

    prob = float(_model.predict_proba(X)[0, 1])
    decision = _apply_policy(prob)

    shap_vals = _explainer.shap_values(X)
    raw = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
    reasons = _top_reasons(raw, _feature_columns)

    latency_ms = (time.perf_counter() - t0) * 1000

    blinded_txn_id = blind_identifier(tx_dict["transaction_id"])
    decision_id = log_decision(
        transaction_id=blinded_txn_id,
        score=prob,
        decision=decision,
        top_reasons=reasons,
        latency_ms=latency_ms,
        model_version=MODEL_VERSION,
        amount=tx_dict["amount"],
        transaction_type=tx_dict["type"],
    )

    return {
        "decision_id": decision_id,
        "transaction_id": tx_dict["transaction_id"],
        "score": round(prob, 6),
        "decision": decision,
        "reasons": reasons,
        "latency_ms": round(latency_ms, 3),
        "model_version": MODEL_VERSION,
    }
