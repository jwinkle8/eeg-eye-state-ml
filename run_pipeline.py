"""End-to-end pipeline: load -> snapshot -> clean -> split -> train -> evaluate -> save.

Usage:
    python run_pipeline.py
"""

from src.data import load_data, clean_data, chronological_split
from src.model import build_model
from src.evaluate import evaluate, save_results


def main():
    load_data()
    X, y, cleaning_info = clean_data()

    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)

    model = build_model()
    model.fit(X_train, y_train)

    results = {
        "cleaning": cleaning_info,
        "validation": evaluate(model, X_val, y_val),
        "test": evaluate(model, X_test, y_test),
    }
    print("validation", results["validation"])
    print("test", results["test"])

    save_results(results)


if __name__ == "__main__":
    main()
