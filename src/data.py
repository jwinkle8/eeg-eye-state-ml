"""Data sourcing and splitting for the EEG Eye State dataset."""

import pandas as pd

from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch

RAW_DATA_PATH = Path("data/raw/eeg_eye_state.parquet")
CLEAN_DATA_PATH = Path("data/clean/eeg_eye_state_clean.parquet")

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


def chronological_split(X, y, val_size: float = 0.2, test_size: float = 0.2):
    """60/20/20 train/validation/test split by position in the recording.

    The samples are a time series, so neighboring rows are nearly identical.
    A shuffled split leaks information; slicing in time order does not.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    n = len(X)
    train_end = int(n * (1 - val_size - test_size))  # end of first 60%
    val_end = int(n * (1 - test_size))               # end of first 80%

    # .iloc[a:b] excludes b, so the three slices touch but never overlap
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    assert len(X_train) + len(X_val) + len(X_test) == n
    assert min(len(X_train), len(X_val), len(X_test)) > 0

    print()
    print("Chronological split")
    print("-------------------")
    for name, part in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"{name}: rows {part.index[0]}-{part.index[-1]} ({len(part)} rows), "
              f"class balance {part.value_counts(normalize=True).round(3).to_dict()}")

    return X_train, X_val, X_test, y_train, y_val, y_test
