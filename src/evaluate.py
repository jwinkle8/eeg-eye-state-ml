"""Evaluation helpers."""


def evaluate(model, X_test, y_test) -> dict:
    """Return test-set metrics for a fitted model.

    Suggested metrics: accuracy, F1, confusion matrix.
    """
    # TODO
    raise NotImplementedError


def save_results(results: dict, path: str = "results/metrics.json") -> None:
    """Write metrics to disk so the README numbers are reproducible."""
    # TODO
    raise NotImplementedError
