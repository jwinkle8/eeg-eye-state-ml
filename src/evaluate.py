"""Evaluation helpers."""

import json
from pathlib import Path

import numpy as np

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


def summarize_seeds(runs: dict) -> dict:
    """Mean and standard deviation of metrics across random-split seeds.

    `runs` maps seed -> {"validation": evaluate(...), "test": evaluate(...)}.
    """
    summary = {"seeds": [int(s) for s in runs], "n_seeds": len(runs)}
    for part in ["validation", "test"]:
        for metric in ["accuracy", "macro_f1"]:
            values = np.array([r[part][metric] for r in runs.values()])
            summary[f"{part}_{metric}_mean"] = float(values.mean())
            summary[f"{part}_{metric}_std"] = float(values.std(ddof=1))
    gaps = [abs(r["validation"]["accuracy"] - r["test"]["accuracy"]) for r in runs.values()]
    summary["max_val_test_accuracy_gap"] = float(max(gaps))
    return summary


def save_results(results: dict, path: str = "results/metrics.json") -> None:
    """Write metrics to disk so the README numbers are reproducible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {path}")
