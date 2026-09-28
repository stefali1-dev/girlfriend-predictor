"""The web form's answers → the model's features, the number, and what it leaned on.

Answers use the model's own names and codes where one exists (age, sex, education, earnings,
...). Three are form-only: `student` (true/false), `work` (none / part_time / full_time) and
`weight_kg`. A missing or null answer means skipped. Only age is required.
"""

import json
import sys
from pathlib import Path

import joblib
from catboost import Pool
from scipy.special import expit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from predict import person_row, predict_proba  # noqa: E402

MODEL = ROOT / "models/partnered.joblib"
TYPICAL = ROOT / "models/typical.json"

NUMBERS = {
    "age": (18, 43),
    "earnings": (0, 1_000_000),
    "height_cm": (120, 230),
    "weight_kg": (30, 250),
    "attendance_worship": (1, 8),  # the survey's scale: 1 never, 3 less than monthly, 6 weekly, 8 daily
    "big5_extraversion": (1, 7),
    "big5_conscientiousness": (1, 7),
    "big5_emotional_stability": (1, 7),
    "nonresident_children": (0, 12),
}
# Weeks and hours a year. Part-time is about 20 hours a week, all year.
WORK = {"none": (0, 0), "part_time": (52, 1040), "full_time": (52, 2080)}
# Each model feature filled from a form answer; the explanation adds features up per answer.
ANSWER_OF = {"weeks_worked": "work", "hours_worked": "work", "bmi": "weight_kg", "enrolled": "student"}
# Skipped answers that get a typical value (skipped choices stay unknown, as in training).
FILLED_WHEN_SKIPPED = ["work", "earnings", "height_cm", "weight_kg", "attendance_worship",
                       "big5_extraversion", "big5_conscientiousness", "big5_emotional_stability",
                       "nonresident_children"]
# How far one answer moved the log-odds, in words. 0.15 is about 4 points near 50%, 0.4 about 10.
SIZES = [(0.4, "a lot"), (0.15, "some"), (0.03, "a little")]

model = joblib.load(MODEL)
typical = json.loads(TYPICAL.read_text())
FORM_CATEGORIES = ["sex", "education", "race_ethnicity", "census_region"]
CHOICES = {field: model["categories"][field] for field in FORM_CATEGORIES}
CHOICES["work"] = ["none", "part_time", "full_time"]


def check(answers):
    """Raise ValueError with a plain message for anything the form should never send."""
    if not isinstance(answers, dict):
        raise ValueError("send a JSON object")
    unknown = set(answers) - set(CHOICES) - set(NUMBERS) - {"student"}
    if unknown:
        raise ValueError(f"unknown fields: {sorted(unknown)}")
    if answers.get("age") is None:
        raise ValueError("age is required")
    if not isinstance(answers["age"], int):
        raise ValueError("age must be a whole number")
    for field, allowed in CHOICES.items():
        if answers.get(field) is not None and answers[field] not in allowed:
            raise ValueError(f"{field} must be one of {allowed}")
    for field, (low, high) in NUMBERS.items():
        value = answers.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f"{field} must be a number from {low} to {high}")
    if answers.get("student") is not None and not isinstance(answers["student"], bool):
        raise ValueError("student must be true or false")


def features(answers):
    """One person's model features. Skipped numbers get the typical value for their sex and age."""
    given = {key: value for key, value in answers.items() if value is not None}
    age = given["age"]
    band = next(f"{low}-{high}" for low, high in [(18, 24), (25, 29), (30, 34), (35, 43)]
                if low <= age <= high)
    person = dict(typical[f"{given.get('sex', 'any')} {band}"])
    person.update({key: value for key, value in given.items()
                   if key not in ("student", "work", "weight_kg")})

    if "work" in given:
        person["weeks_worked"], person["hours_worked"] = WORK[given["work"]]
    if "weight_kg" in given:
        person["bmi"] = given["weight_kg"] / (person["height_cm"] / 100) ** 2
    if "student" in given:
        if not given["student"]:
            person["enrolled"] = "not_enrolled"
        elif given.get("education") == "bachelor_plus":
            person["enrolled"] = "graduate"
        else:
            person["enrolled"] = "college_4yr"
    return person_row(model, person)


def leaned_on(row, given, answer_of):
    """Up to 4 given answers that moved this person's number most, compared with the average person.

    CatBoost's SHAP values split the model's log-odds exactly; the calibration step only
    scales them, so they add up to the final number's log-odds too.
    """
    pipe = model["base"]["catboost"]
    catboost = pipe[-1]
    shap = catboost.get_feature_importance(
        Pool(pipe[:-1].transform(row), cat_features=list(catboost.get_params()["cat_features"])),
        type="ShapValues")[0]
    scale = model["stacker"][-1].coef_[0, 0]
    per_answer = {}
    for feature, value in zip(pipe[:-1].get_feature_names_out(), shap[:-1] * scale):
        answer = answer_of.get(feature, feature)
        if answer in given:
            per_answer[answer] = per_answer.get(answer, 0.0) + value
    average = float(expit(shap[-1] * scale + model["stacker"][-1].intercept_[0]))

    rows = []
    for answer, value in sorted(per_answer.items(), key=lambda kv: -abs(kv[1]))[:4]:
        size = next((word for limit, word in SIZES if abs(value) >= limit), None)
        if size:
            rows.append({"answer": answer, "direction": "up" if value > 0 else "down", "size": size})
    return average, rows


def answer(answers):
    check(answers)
    answer_of = ANSWER_OF
    if answers.get("work") == "none" and answers.get("earnings") is None:
        # No job means no pay. The pay slider is hidden then, so it counts as part of the "work" answer.
        answers = {**answers, "earnings": 0}
        answer_of = {**ANSWER_OF, "earnings": "work"}
    given = {key for key, value in answers.items() if value is not None}
    row = features(answers)
    average, rows = leaned_on(row, given, answer_of)
    return {
        "probability": round(float(predict_proba(model, row)[0]), 4),
        "average": round(average, 4),
        "leaned_on": rows,
        "assumed": [field for field in FILLED_WHEN_SKIPPED
                    if field not in given],
    }
