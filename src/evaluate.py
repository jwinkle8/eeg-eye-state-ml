"""Evaluation helpers."""

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate(model, X, y) -> dict:
    """Return metrics for a fitted model on one split (validation or test).

    Labels: 0 = eyes open, 1 = eyes closed.
    All values are plain Python types so the dict is JSON-serializable.
    """
    y_pred = model.predict(X)

    return {
        "n_rows": len(y),
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
        # Rows = true label, columns = predicted label: [[TN, FP], [FN, TP]]
        # labels=[0, 1] keeps it 2x2 even if one class is never predicted
        "confusion_matrix": confusion_matrix(y, y_pred, labels=[0, 1]).tolist(),
        # Accuracy of always guessing this split's most common class
        "majority_class_share": float(y.value_counts(normalize=True).max()),
    }


def save_results(results: dict, path: str = "results/metrics.json") -> None:
    """Write metrics to disk so the README numbers are reproducible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {path}")
