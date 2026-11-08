"""Data sourcing and splitting for the EEG Eye State dataset."""

import pandas as pd

RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Download the EEG Eye State dataset from OpenML.

    Returns:
        X: 14 EEG channel readings per sample.
        y: eye state label (0 = open, 1 = closed).
    """
    # TODO: fetch with sklearn.datasets.fetch_openml(data_id=1471, as_frame=True)
    # TODO: inspect column names and label values before mapping them
    raise NotImplementedError


def save_raw_snapshot(X: pd.DataFrame, y: pd.Series, path: str = "data/raw/eeg_eye_state.parquet") -> None:
    """Save the untouched download to Parquet before any cleaning.

    Keeps a versioned copy of exactly what was trained on, so results can be
    reproduced even if the source changes or goes offline.
    """
    # TODO: combine X and y into one DataFrame and write with df.to_parquet(path)
    raise NotImplementedError


def clean_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Basic cleaning.

    The raw recording has a few extreme spikes (sensor artifacts).
    """
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
