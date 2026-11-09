"""Data sourcing and splitting for the EEG Eye State dataset."""

import pandas as pd

from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

RAW_DATA_PATH = Path("data/raw/eeg_eye_state.parquet")
CLEAN_DATA_PATH = Path("data/clean/eeg_eye_state_clean.parquet")

RANDOM_STATE = 86  # seed for the random split

# Feasibility thresholds
OUTLIER_LOWER_THRESHOLD = 2000  # uV
OUTLIER_UPPER_THRESHOLD = 6500  # uV

def load_data():
    """Ensure the raw snapshot exists (fetch from OpenML), then print a quick inspection.
    """
    if not RAW_DATA_PATH.exists():
        save_raw_snapshot(fetch_openml(data_id=1471, as_frame=True))

    eeg_data_df = pd.read_parquet(RAW_DATA_PATH)
    print("Load Data")
    print("---------")
    print()
    print(f"Shape: {eeg_data_df.shape}")
    print(f"dtypes: {eeg_data_df.dtypes}")
    print(f"Eye state counts: {eeg_data_df['Class'].value_counts()}")

def save_raw_snapshot(eeg_data_bunch: Bunch) -> None:
    """Save the untouched download to Parquet before any cleaning.

    Keeps a versioned copy of exactly what was trained on, so results can be
    reproduced even if the source changes or goes offline.
    """
    df = eeg_data_bunch.frame

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_DATA_PATH, engine="pyarrow", compression="snappy")


def clean_data() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Basic cleaning.

    The raw recording has a few extreme spikes (sensor artifacts).
    """
    eeg_data_df = pd.read_parquet(RAW_DATA_PATH)

    print()
    print("Data cleaning")
    print("-------------")
    print()
    # Ensure data matches OpenML claims
    assert(eeg_data_df.shape == (14980, 15))
    for i in range(1, 15):
        assert(f"V{i}" in eeg_data_df.columns)
    print("Shape/column name asserts passed!")

    # Ensure no missing values (see "Missing Values" column on dataset source site "Variables Table")
    assert(eeg_data_df.isna().sum().sum() == 0)

    # Map column names to channel names (see "Variable Names" column on dataset source site "Variables Table")
    new_channel_names = [
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
    ]
    column_mapping = dict(zip([f"V{i}" for i in range(1, 15)], new_channel_names))
    eeg_data_df = eeg_data_df.rename(columns=column_mapping)

    # Map target to 0/1: raw 1 = eyes open -> 0, raw 2 = eyes closed -> 1
    eeg_data_df["Class"] = eeg_data_df["Class"].astype(int).map({1: 0, 2: 1})
    assert eeg_data_df["Class"].isna().sum() == 0  # fails if an unexpected label appears

    # Outlier detection
    # Upper and lower outlier thresholds were chosen by reviewing the spread of the data
    print("Channel summary statistics")
    print(eeg_data_df[new_channel_names].agg(["min", "median", "max"]).T)

    # Flag observations outside the chosen outlier range
    channels = eeg_data_df[new_channel_names]
    is_outlier = (
        (channels <= OUTLIER_LOWER_THRESHOLD) | (channels > OUTLIER_UPPER_THRESHOLD)
    ).any(axis=1)
    flagged = eeg_data_df.index[is_outlier]
    print(f"{len(flagged)} rows outside plausibility range of [{OUTLIER_LOWER_THRESHOLD}, {OUTLIER_UPPER_THRESHOLD}]")

    
    n_rows = len(eeg_data_df)
    cleaning_info = {
        "rows_before": n_rows,
        "outlier_upper_threshold": OUTLIER_UPPER_THRESHOLD,
        "outlier_lower_threshold": OUTLIER_LOWER_THRESHOLD,
        "dropped_row_indices": flagged.tolist(),  # positions in the raw file (lineage)
        "rows_dropped": int(is_outlier.sum()),
    }
    eeg_data_df = eeg_data_df.loc[~is_outlier].reset_index(drop=True)
    cleaning_info["rows_after"] = len(eeg_data_df)

    # Save the cleaned data separately; the raw snapshot is never overwritten
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    eeg_data_df.to_parquet(CLEAN_DATA_PATH, engine="pyarrow", compression="snappy", index=False)
    print(f"Saved cleaned data to {CLEAN_DATA_PATH}")

    X =eeg_data_df.drop(columns="Class")
    y = eeg_data_df["Class"]
    return X, y, cleaning_info


def random_split(X, y, val_size: float = 0.2, test_size: float = 0.2):
    """60/20/20 train/validation/test split, shuffled and stratified by class.

    Caveat: the rows are a ~128 Hz time series, so most test rows have a
    near-identical neighbor (~8 ms away) in train. Scores from this split
    are optimistic; they do not measure performance on new recordings.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, split_info
    """
    # Step 1: hold out the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    # Step 2: split the remainder; val fraction is relative to what's left
    # (0.2 / 0.8 = 0.25 of the remainder = 20% of the total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size / (1 - test_size),
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    assert len(X_train) + len(X_val) + len(X_test) == len(X)

    # Record size and class balance of each part (printed and saved to metrics.json)
    print()
    print("Random split")
    print("------------")
    split_info = {"random_state": RANDOM_STATE}
    for name, part in [("train", y_train), ("val", y_val), ("test", y_test)]:
        balance = part.value_counts(normalize=True)
        split_info[name] = {
            "n_rows": len(part),
            # Cast numpy keys/values to plain int/float so json.dump accepts them
            "class_balance": {int(k): float(v) for k, v in balance.items()},
        }
        print(f"{name}: {len(part)} rows, class balance {balance.round(3).to_dict()}")

    return X_train, X_val, X_test, y_train, y_val, y_test, split_info
