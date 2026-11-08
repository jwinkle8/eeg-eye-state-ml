"""Model definitions."""

RANDOM_STATE = 42


def get_models() -> dict:
    """Return the models to compare, keyed by name.

    Suggested:
        - baseline: DummyClassifier(strategy="most_frequent")
        - logistic_regression: StandardScaler + LogisticRegression in a sklearn Pipeline
        - random_forest: RandomForestClassifier(random_state=RANDOM_STATE)
    """
    # TODO
    raise NotImplementedError
