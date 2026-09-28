"""The outside-check helpers: value groups, direction lines, and the any-partner table."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from outside_check import (TARGET, any_partner_table, direction, full_model_reference,  # noqa: E402
                           points, transported, value_groups)


class StubModel:
    """Stands in for a fitted pipeline: predict_proba returns one constant."""

    def __init__(self, value):
        self.value = value

    def predict_proba(self, X):
        v = np.full(len(X), self.value)
        return np.column_stack([1 - v, v])


def test_transportled_scores_missing_rows_with_the_five_feature_model():
    frame = pd.DataFrame({
        "age": [30, 30, 30], "sex": ["male"] * 3, "race_ethnicity": ["other"] * 3,
        "education": ["hs"] * 3, "employed": [1.0] * 3,
        "height_cm": [180.0, np.nan, np.nan], "bmi": [24.0, 25.0, np.nan],
    })
    seven, five = StubModel(0.8), StubModel(0.4)
    # both measured -> seven features; either one missing -> five features
    assert transported(seven, five, frame).tolist() == [0.8, 0.4, 0.4]
    # HCMST's situation: nothing measured, the seven-feature model is never called
    all_missing = frame.assign(height_cm=np.nan, bmi=np.nan)
    assert transported(seven, five, all_missing).tolist() == [0.4, 0.4, 0.4]


def test_value_groups_use_comparable_bins():
    bands = value_groups(pd.Series([18, 24, 25, 34, 43, 44], name="age"))
    assert bands.tolist()[:5] == ["18-24", "18-24", "25-29", "30-34", "35-43"]
    assert pd.isna(bands.tolist()[5])  # outside the model's age range: no band

    employed = value_groups(pd.Series([0, 1, pd.NA], name="employed", dtype="Int64"))
    assert employed[:2].tolist() == ["not employed", "employed"]
    assert employed.isna()[2]

    tertiles = value_groups(pd.Series([1.0, 2.0, 3.0, np.nan], name="bmi"))
    assert set(tertiles.dropna()) == {"low", "middle", "high"}

    race = value_groups(pd.Series(["black", "hispanic"], name="race_ethnicity"))
    assert race.tolist() == ["black", "hispanic"]


def test_points_translate_log_odds_into_percentage_points():
    values = pd.Series(["a", "a", "b", "b"], name="race_ethnicity")
    sv = pd.Series([-0.5, -0.5, 0.25, 0.25])
    got = points(sv, values, base_rate=0.4)
    assert set(got) == {"a", "b"}
    assert got["a"] < 0 < got["b"]  # a negative log-odds push pulls below the 40% base rate
    assert abs(got["b"] - (expit(logit(0.4) + 0.25) - 0.4) * 100) < 1e-9


def test_direction_reports_extremes_and_skips_unmeasured_features():
    values = pd.Series(["less_than_hs", "hs", "some_college", "bachelor_plus"] * 2,
                       name="education")
    sv = pd.Series([2.0, 0.0, -1.0, 0.5] * 2)
    line = direction(sv, values, 0.5)
    assert line.startswith("some_college -") and "less_than_hs +" in line

    empty = pd.Series(np.nan, index=range(3), name="height_cm")  # HCMST's situation
    assert direction(pd.Series(0.0, index=range(3)), empty, 0.5) == "not measured in this survey"


def test_full_model_reference_reads_the_committed_metrics():
    log_loss, auc = full_model_reference()
    assert 0 < log_loss < 1 and 0.5 < auc < 1


def test_any_partner_table_counts_only_the_not_partnered():
    hcmst = pd.DataFrame({
        "age": [20, 22, 23, 36, 41, 37, 25],
        "sex": ["male", "male", "female", "female", "female", "male", "male"],
        TARGET: [0, 0, 0, 0, 0, 0, 1],          # the partnered row must be left out
        "any_partner": [1, 0, 1, 1, 1, 0, 1],
    })
    got = any_partner_table(hcmst)
    band_18_24 = got[got["age"] == "18-24"].iloc[0]
    assert band_18_24["men"] == "2" and band_18_24["men with a partner"] == "50%"
    assert band_18_24["women"] == "1" and band_18_24["women with a partner"] == "100%"
    all_ = got[got["age"] == "all 18-43"].iloc[0]
    assert all_["men with a partner"] == "33%"  # 1 of 3 not-partnered men
    assert len(got) == 3                        # 18-24, 35-43 and the total row
