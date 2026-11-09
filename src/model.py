"""Model definition."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def build_model() -> Pipeline:
    """Standardize the channels, then fit logistic regression.

    The scaler lives inside the Pipeline, so it learns its mean and std
    from the training rows only when the pipeline is fit.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
