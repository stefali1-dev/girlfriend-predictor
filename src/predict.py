"""Probability that one person is partnered (married or living together).

Usage: python src/predict.py person.json    (or pipe the JSON on stdin)

The JSON holds any of the model's features; only age is required. Missing fields
are treated as unknown, the same way missing survey answers were in training.
Example: {"age": 29, "sex": "male", "education": "bachelor_plus", "earnings": 55000}
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL = Path("models/partnered.joblib")


def base_probabilities(model, people):
    """One column per base model: its probability for each row of `people`."""
    return np.column_stack([m.predict_proba(people[model["features"]])[:, 1]
                            for m in model["base"].values()])


def predict_proba(model, people):
    return model["stacker"].predict_proba(base_probabilities(model, people))[:, 1]


def person_row(model, person):
    """Check the JSON fields and turn them into a one-row table with every feature column."""
    unknown = set(person) - set(model["features"])
    if unknown:
        sys.exit(f"unknown fields {sorted(unknown)}; allowed: {model['features']}")
    # A JSON null means "unknown", exactly like leaving the field out.
    person = {col: value for col, value in person.items() if value is not None}
    if "age" not in person:
        sys.exit("age is required: it drives most of the prediction")

    categorical = list(model["categories"])
    numeric = [col for col in model["features"] if col not in categorical]
    for col, allowed in model["categories"].items():
        if col in person and person[col] not in allowed:
            sys.exit(f"{col} must be one of {allowed}, got {person[col]!r}")
    for col in numeric:
        if col in person and not isinstance(person[col], (int, float)):
            sys.exit(f"{col} must be a number, got {person[col]!r}")

    row = pd.DataFrame([person]).reindex(columns=model["features"])
    # Missing categories must stay NaN (not the text "nan") so the pipelines impute them as in
    # training; pandas 3 keeps NaN through astype("str"), pandas 2 would not.
    row[categorical] = row[categorical].astype("str")
    row[numeric] = row[numeric].astype(float)
    return row


def main():
    person = json.load(open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin)
    model = joblib.load(MODEL)
    print(f"{predict_proba(model, person_row(model, person))[0]:.3f}")


if __name__ == "__main__":
    main()
