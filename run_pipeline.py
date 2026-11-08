"""End-to-end pipeline: load -> snapshot -> clean -> split -> train -> evaluate -> save.

Usage:
    python run_pipeline.py
"""

from src.data import load_data, clean_data, random_split, chronological_split
from src.model import get_models
from src.evaluate import evaluate, save_results


def main():
    load_data()
    X, y = clean_data()

    results = {}
    for split_name, split_fn in [("random", random_split), ("chronological", chronological_split)]:
        X_train, X_test, y_train, y_test = split_fn(X, y)
        for model_name, model in get_models().items():
            model.fit(X_train, y_train)
            results[f"{split_name}/{model_name}"] = evaluate(model, X_test, y_test)
            print(split_name, model_name, results[f"{split_name}/{model_name}"])

    save_results(results)


if __name__ == "__main__":
    main()
