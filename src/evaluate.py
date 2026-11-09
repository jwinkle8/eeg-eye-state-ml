"""Evaluation helpers."""


def evaluate(model, X, y) -> dict:
    """Return metrics for a fitted model on one split (validation or test).

    Suggested metrics: accuracy, macro F1, confusion matrix.
    Convert numpy values to plain floats/lists so the dict is JSON-serializable.
    """
    # TODO
    raise NotImplementedError


def save_results(results: dict, path: str = "results/metrics.json") -> None:
    """Write metrics to disk so the README numbers are reproducible."""
    # TODO
    raise NotImplementedError
