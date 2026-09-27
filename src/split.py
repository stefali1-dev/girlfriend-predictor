"""Split NLSY97 into train and test by person: 20% of people go to test, all their rows.

Why by person and not by row: each person has up to 21 rows (one per survey round),
and their answers barely change between rounds. With a row split, a person's 2010 row
could be in train and their 2012 row in test; the model would partly "recognise" them
and the test score would look better than it will be on new people.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CLEAN = Path("data/clean/nlsy97.parquet")
OUT_DIR = Path("data/curated")
SEED = 42
TEST_SHARE = 0.2
MIN_AGE, MAX_AGE = 18, 43  # NLSY97 covers these ages for every birth cohort (1980-84)


def main():
    df = pd.read_parquet(CLEAN)
    df = df[df["age"].between(MIN_AGE, MAX_AGE)]

    people = df["person_id"].unique()
    rng = np.random.default_rng(SEED)
    test_people = rng.choice(people, size=round(TEST_SHARE * len(people)), replace=False)
    is_test = df["person_id"].isin(test_people)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df[~is_test].to_parquet(OUT_DIR / "train.parquet", index=False)
    df[is_test].to_parquet(OUT_DIR / "test.parquet", index=False)

    for name, part in [("train", df[~is_test]), ("test", df[is_test])]:
        print(f"{name}: {len(part):,} rows, {part['person_id'].nunique():,} people, "
              f"partnered {part['partnered'].mean():.3f}")


if __name__ == "__main__":
    main()
