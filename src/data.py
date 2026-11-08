"""Data sourcing and splitting for the EEG Eye State dataset."""

import pandas as pd

from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch

RANDOM_STATE = 42
RAW_DATA_PATH = Path("data/raw/eeg_eye_state.parquet")

def load_data():
    """Ensure the raw snapshot exists (fetch from OpenML), then print a quick inspection.
    """
    if RAW_DATA_PATH.exists():
        print("Using cached dataset...")
    else:
        save_raw_snapshot(fetch_openml(data_id=1471, as_frame=True))

    eeg_data_bunch = pd.read_parquet(RAW_DATA_PATH)
    print("Load Data")
    print("---------")
    print()
    print(eeg_data_bunch.shape)
    print(eeg_data_bunch.dtypes)
    print(eeg_data_bunch["Class"].value_counts())

def save_raw_snapshot(eeg_data_bunch: Bunch) -> None:
    """Save the untouched download to Parquet before any cleaning.

    Keeps a versioned copy of exactly what was trained on, so results can be
    reproduced even if the source changes or goes offline.
    """
    df = eeg_data_bunch.frame

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_DATA_PATH, engine="pyarrow", compression="snappy")


def clean_data() -> tuple[pd.DataFrame, pd.Series]:
    """Basic cleaning.

    The raw recording has a few extreme spikes (sensor artifacts).
    """
    # TODO: read from "data/raw/eeg_eye_state.parquet"
    # TODO: check for missing values
    # TODO: decide how to handle outliers (drop, clip, or leave in) and note it in README
    raise NotImplementedError


def random_split(X, y, test_size: float = 0.2):
    """Shuffled, stratified train/test split."""
    # TODO: sklearn.model_selection.train_test_split(..., stratify=y, random_state=RANDOM_STATE)
    raise NotImplementedError


def chronological_split(X, y, test_size: float = 0.2):
    """Train on the first part of the recording, test on the last part.

    The samples are a time series, so neighboring rows are nearly identical.
    A shuffled split leaks information; this split does not.
    """
    # TODO: slice by row position (no shuffling)
    raise NotImplementedError
