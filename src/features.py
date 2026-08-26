import numpy as np
import pandas as pd

TYPE_ENCODING = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}
AMOUNT_DRAIN_RATIO = 0.9
HIGH_AMOUNT_THRESHOLD = 200_000
VELOCITY_WINDOW = 1

FEATURE_COLUMNS = [
    "balance_error_orig",
    "balance_error_dest",
    "drain_flag",
    "zero_orig_after",
    "type_encoded",
    "amount_log",
    "orig_txn_count_1h",
    "orig_txn_sum_1h",
    "dest_in_degree_1h",
    "high_amount_flag",
]


# Computes the accounting residual on the origin account; near-zero for legit transactions
def compute_balance_error_orig(df: pd.DataFrame) -> pd.Series:
    return df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]


# Computes the accounting residual on the destination account
def compute_balance_error_dest(df: pd.DataFrame) -> pd.Series:
    return df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]


# Returns 1 if amount drains at least 90% of a non-zero origin balance
def compute_drain_flag(df: pd.DataFrame) -> pd.Series:
    return ((df["oldbalanceOrg"] > 0) & (df["amount"] >= AMOUNT_DRAIN_RATIO * df["oldbalanceOrg"])).astype(int)


# Returns 1 if the origin account balance is zero after the transaction
def compute_zero_orig_after(df: pd.DataFrame) -> pd.Series:
    return (df["newbalanceOrig"] == 0).astype(int)


# Maps transaction type to a fixed integer encoding consistent across train and inference
def compute_type_encoded(df: pd.DataFrame) -> pd.Series:
    return df["type"].map(TYPE_ENCODING).fillna(-1).astype(int)


# Applies log1p transformation to the transaction amount to reduce skew
def compute_amount_log(df: pd.DataFrame) -> pd.Series:
    return np.log1p(df["amount"])


# Counts transactions from the same origin account within the previous 1 step
def compute_orig_txn_count_1h(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(0, index=df.index, dtype=int)
    for idx, row in df.iterrows():
        window = df[
            (df["nameOrig"] == row["nameOrig"])
            & (df["step"] >= row["step"] - VELOCITY_WINDOW)
            & (df["step"] < row["step"])
        ]
        result.at[idx] = len(window)
    return result


# Sums transaction amounts from the same origin account within the previous 1 step
def compute_orig_txn_sum_1h(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(0.0, index=df.index, dtype=float)
    for idx, row in df.iterrows():
        window = df[
            (df["nameOrig"] == row["nameOrig"])
            & (df["step"] >= row["step"] - VELOCITY_WINDOW)
            & (df["step"] < row["step"])
        ]
        result.at[idx] = window["amount"].sum()
    return result


# Counts distinct origin senders to the same destination within the previous 1 step (mule-account burst detection)
def compute_dest_in_degree_1h(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(0, index=df.index, dtype=int)
    for idx, row in df.iterrows():
        window = df[
            (df["nameDest"] == row["nameDest"])
            & (df["step"] >= row["step"] - VELOCITY_WINDOW)
            & (df["step"] < row["step"])
        ]
        result.at[idx] = window["nameOrig"].nunique()
    return result


# Returns 1 if the transaction amount exceeds the high-value threshold
def compute_high_amount_flag(df: pd.DataFrame) -> pd.Series:
    return (df["amount"] > HIGH_AMOUNT_THRESHOLD).astype(int)


# Vectorised helper that computes velocity and graph features using groupby for performance
def _compute_velocity_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_idx"] = np.arange(len(df))

    orig_count = []
    orig_sum = []
    dest_indegree = []

    step_arr = df["step"].values
    orig_arr = df["nameOrig"].values
    dest_arr = df["nameDest"].values
    amount_arr = df["amount"].values

    for i in range(len(df)):
        s = step_arr[i]
        mask_window = (step_arr >= s - VELOCITY_WINDOW) & (step_arr < s)

        orig_mask = mask_window & (orig_arr == orig_arr[i])
        orig_count.append(int(orig_mask.sum()))
        orig_sum.append(float(amount_arr[orig_mask].sum()))

        dest_mask = mask_window & (dest_arr == dest_arr[i])
        dest_indegree.append(int(np.unique(orig_arr[dest_mask]).shape[0]))

    df["orig_txn_count_1h"] = orig_count
    df["orig_txn_sum_1h"] = orig_sum
    df["dest_in_degree_1h"] = dest_indegree
    return df


# Adds all 10 engineered features to the input DataFrame and returns it
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["balance_error_orig"] = compute_balance_error_orig(df)
    df["balance_error_dest"] = compute_balance_error_dest(df)
    df["drain_flag"] = compute_drain_flag(df)
    df["zero_orig_after"] = compute_zero_orig_after(df)
    df["type_encoded"] = compute_type_encoded(df)
    df["amount_log"] = compute_amount_log(df)
    df["high_amount_flag"] = compute_high_amount_flag(df)
    df = _compute_velocity_features_fast(df)
    return df


# Returns the ordered list of feature column names used for model training and inference
def get_feature_columns() -> list[str]:
    return FEATURE_COLUMNS
