"""Typical value of every numeric feature for each sex and age band: models/typical.json.

The web form fills skipped numbers with these. CatBoost reads a missing number as "below
everyone" (a blank height counts as very short), so a blank is not a neutral answer.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

TRAIN = Path("data/curated/train.parquet")
MODEL = Path("models/partnered.joblib")
OUT = Path("models/typical.json")
BANDS = [(18, 24), (25, 29), (30, 34), (35, 43)]


def main():
    train = pd.read_parquet(TRAIN)
    model = joblib.load(MODEL)
    numeric = [col for col in model["features"] if col not in model["categories"] and col != "age"]
    typical = {}
    for sex in ["female", "male", "any"]:
        for low, high in BANDS:
            rows = train[((train["sex"] == sex) | (sex == "any")) & train["age"].between(low, high)]
            typical[f"{sex} {low}-{high}"] = {col: float(rows[col].median()) for col in numeric}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(typical, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
