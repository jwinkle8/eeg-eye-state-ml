"""End-to-end pipeline: load -> snapshot -> clean -> split -> train -> evaluate -> save.

Usage:
    python run_pipeline.py                      # random split (default)
    python run_pipeline.py --split chronological

Random split: a detailed run with the fixed seed (RANDOM_STATE), plus a sweep
over SWEEP_SEEDS to show how much the scores depend on which rows land in
validation and test. Results go to results/metrics.json.

Chronological split: reproduces the earlier chronological results.
Results go to results/metrics_chronological.json.
"""

import argparse

from src.data import load_data, clean_data, random_split, chronological_split
from src.model import build_model
from src.evaluate import evaluate, save_results, summarize_seeds

SWEEP_SEEDS = range(30)  # seeds 0-29 for the random-split sweep

OUTPUT_PATHS = {
    "random": "results/metrics.json",
    "chronological": "results/metrics_chronological.json",
}


def fit_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test) -> dict:
    """Fit a fresh model on train; return validation and test metrics."""
    model = build_model()
    model.fit(X_train, y_train)
    return {
        "validation": evaluate(model, X_val, y_val),
        "test": evaluate(model, X_test, y_test),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=["random", "chronological"], default="random")
    args = parser.parse_args()

    load_data()
    X, y, cleaning_info = clean_data()

    split_fn = random_split if args.split == "random" else chronological_split
    *parts, split_info = split_fn(X, y)

    results = {"cleaning": cleaning_info, "split": split_info}
    results.update(fit_and_evaluate(*parts))
    print("validation", results["validation"])
    print("test", results["test"])

    if args.split == "random":
        runs = {}
        for seed in SWEEP_SEEDS:
            *parts, _ = random_split(X, y, random_state=seed, verbose=False)
            runs[seed] = fit_and_evaluate(*parts)
        results["seed_sweep"] = summarize_seeds(runs)
        print("seed sweep", results["seed_sweep"])

    save_results(results, OUTPUT_PATHS[args.split])


if __name__ == "__main__":
    main()
