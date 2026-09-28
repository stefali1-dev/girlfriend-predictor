"""The web form's model code: needs a trained model and models/typical.json (make all)."""

import sys
from pathlib import Path

import pytest
from catboost import Pool
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sagemaker"))
import form  # noqa: E402


def test_skipped_number_is_the_typical_value_not_blank():
    height = form.typical["male 30-34"]["height_cm"]
    skipped = form.answer({"age": 31, "sex": "male"})
    given = form.answer({"age": 31, "sex": "male", "height_cm": height})
    assert skipped["probability"] == given["probability"]
    assert "height_cm" in skipped["assumed"] and "height_cm" not in given["assumed"]


def test_null_is_skipped():
    assert form.answer({"age": 25, "earnings": None}) == form.answer({"age": 25})


def test_no_work_and_no_pay_means_zero_earnings_counted_under_work():
    result = form.answer({"age": 30, "sex": "female", "work": "none"})
    explicit = form.answer({"age": 30, "sex": "female", "work": "none", "earnings": 0})
    assert result["probability"] == explicit["probability"]
    assert "earnings" not in result["assumed"]
    assert {row["answer"] for row in result["leaned_on"]} <= {"age", "sex", "work"}


def test_weight_uses_the_given_height():
    row = form.features({"age": 30, "height_cm": 200, "weight_kg": 80})
    assert row["bmi"].iloc[0] == pytest.approx(20.0)


def test_explanation_adds_up_to_the_number_and_lists_only_given_answers():
    answers = {"age": 29, "sex": "male", "work": "full_time", "earnings": 55000, "weight_kg": 80}
    row = form.features(answers)
    pipe = form.model["base"]["catboost"]
    shap = pipe[-1].get_feature_importance(
        Pool(pipe[:-1].transform(row), cat_features=list(pipe[-1].get_params()["cat_features"])),
        type="ShapValues")[0]
    platt = form.model["stacker"][-1]
    result = form.answer(answers)
    assert expit(shap.sum() * platt.coef_[0, 0] + platt.intercept_[0]) == pytest.approx(
        result["probability"], abs=1e-4)
    assert {row["answer"] for row in result["leaned_on"]} <= set(answers)
    assert 1 <= len(result["leaned_on"]) <= 4


@pytest.mark.parametrize("bad", [
    {}, {"sex": "male"}, {"age": 17}, {"age": 44}, {"age": 24.5}, {"age": "30"}, {"age": True},
    {"age": 30, "sex": "man"}, {"age": 30, "student": "yes"}, {"age": 30, "shoe_size": 42},
    {"age": 30, "earnings": -1}, [30],
])
def test_bad_answers_are_refused(bad):
    with pytest.raises(ValueError):
        form.check(bad)
