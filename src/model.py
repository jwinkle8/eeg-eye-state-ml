"""Model definition."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 86


def build_model() -> Pipeline:
    """Standardize the channels, then fit logistic regression.

    The scaler lives inside the Pipeline, so it learns its mean and std
    from the training rows only when the pipeline is fit.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        # class_weight="balanced": train is ~61% closed / 39% open; reweight so
        # neither class dominates the loss just by being more frequent
        ("logreg", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])
