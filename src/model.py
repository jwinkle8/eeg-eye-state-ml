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
        # class_weight="balanced": weight each class inversely to its frequency in
        # train, so neither dominates the loss just by being more common. Chosen on
        # the chronological split's validation set; on the random split the
        # classes are ~55% open / 45% closed, so the reweighting is mild.
        ("logreg", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])
