import os
import pandas as pd

RAW_PATH = os.getenv("RAW_DATA_PATH", "data/PaySim.csv")
TRAIN_PATH = os.getenv("TRAIN_PATH", "data/train.parquet")
TEST_PATH = os.getenv("TEST_PATH", "data/test.parquet")

LEGIT_SAMPLE = 500_000
RANDOM_STATE = 42
TRAIN_RATIO = 0.8


# Loads PaySim CSV or falls back to sample data if full dataset is not present
def load_data() -> pd.DataFrame:
    if os.path.exists(RAW_PATH):
        print(f"Loading full dataset from {RAW_PATH}")
        df = pd.read_csv(RAW_PATH)
    elif os.path.exists("data/sample_transactions.csv"):
        print("WARNING: PaySim.csv not found. Using sample_transactions.csv (20 rows).")
        print("Download full dataset from https://www.kaggle.com/datasets/mtalaltariq/paysim-data")
        df = pd.read_csv("data/sample_transactions.csv")
    else:
        raise FileNotFoundError(
            "No data found. Place PaySim.csv at data/PaySim.csv or "
            "use data/sample_transactions.csv for a quick demo."
        )
    return df


# Subsamples all fraud rows plus a fixed number of random legit rows
def subsample(df: pd.DataFrame, legit_n: int, random_state: int) -> pd.DataFrame:
    fraud = df[df["isFraud"] == 1]
    legit_df = df[df["isFraud"] == 0]
    n_samples = min(legit_n, len(legit_df))
    legit = legit_df.sample(n=n_samples, random_state=random_state)
    return pd.concat([fraud, legit], ignore_index=True)


# Sorts the DataFrame by the step column to enforce temporal ordering
def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("step").reset_index(drop=True)


# Splits a time-ordered DataFrame into train and test without shuffling
def time_split(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = int(len(df) * train_ratio)
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


# Saves a DataFrame to parquet, creating the parent directory if needed
def save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)


# Prints a human-readable split summary including fraud rates and step range
def print_summary(raw: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> None:
    total = len(train) + len(test)
    train_fraud_rate = train["isFraud"].mean() * 100
    test_fraud_rate = test["isFraud"].mean() * 100
    step_min = int(raw["step"].min())
    step_max = int(raw["step"].max())

    print(f"{'='*50}")
    print(f"RazorSentry — Data Loader Summary")
    print(f"{'='*50}")
    print(f"Raw CSV rows          : {len(raw):,}")
    print(f"After subsampling     : {total:,}")
    print(f"  Fraud rows          : {int(train['isFraud'].sum() + test['isFraud'].sum()):,}")
    print(f"  Legit rows          : {int((1 - train['isFraud']).sum() + (1 - test['isFraud']).sum()):,}")
    print(f"Train rows            : {len(train):,}")
    print(f"  Train fraud count   : {int(train['isFraud'].sum()):,}")
    print(f"  Train fraud rate    : {train_fraud_rate:.4f}%")
    print(f"Test rows             : {len(test):,}")
    print(f"  Test fraud count    : {int(test['isFraud'].sum()):,}")
    print(f"  Test fraud rate     : {test_fraud_rate:.4f}%")
    print(f"Step range (raw)      : {step_min} — {step_max}")
    print(f"Train saved to        : {TRAIN_PATH}")
    print(f"Test saved to         : {TEST_PATH}")
    print(f"{'='*50}")


# Orchestrates loading, subsampling, sorting, splitting, saving, and printing
def main() -> None:
    raw = load_data()

    print(f"Subsampling: all fraud + {LEGIT_SAMPLE:,} legit rows ...")
    sampled = subsample(raw, LEGIT_SAMPLE, RANDOM_STATE)

    print("Sorting by step (temporal order) ...")
    sampled = sort_by_time(sampled)

    print(f"Splitting: first {int(TRAIN_RATIO*100)}% train, last {int((1-TRAIN_RATIO)*100)}% test (no shuffle) ...")
    train, test = time_split(sampled, TRAIN_RATIO)

    save_parquet(train, TRAIN_PATH)
    save_parquet(test, TEST_PATH)

    print_summary(raw, train, test)


if __name__ == "__main__":
    main()
